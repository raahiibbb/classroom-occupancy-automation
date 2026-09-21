/*
  ESP-2 -- occupancy controller.

  An ESP32 NodeMCU that sits between the camera board and the room's wiring. It
  has three jobs:

    1. Receive the 3x3 occupancy grid that ESP-1 broadcasts over ESP-NOW.
    2. Smooth those readings and switch the four fans and five bulbs, one
       appliance per zone.
    3. Decide when ESP-1 should be running at all, using a PIR motion sensor
       and its own view of how long the room has been empty.

  Job 3 lives here rather than on ESP-1 for two reasons. The PIR sensor is
  wired to this board, and more importantly this is where the smoothed, flicker
  free view of the room already exists -- deciding "the room is empty" from the
  raw per-frame output would be unreliable, because single frames flip state
  fairly often even when nothing is moving.

  Board settings: "ESP32 Dev Module" (or "NodeMCU-32S"), 115200 baud.
*/

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "occupancy_link.h"

// ===========================================================================
// Appliance wiring
// ===========================================================================
// One appliance per zone. The relay board is mostly active-LOW -- pulling the
// input low energises the relay -- except for the last channel, which is
// wired the other way round. Storing the polarity per appliance means the rest
// of the code can just say "on" or "off" and not care.
//
// GPIO 12 is deliberately not used. On this chip it is a boot strapping pin
// that selects the flash voltage, and a relay input holding it high stops the
// board starting at all.
struct Appliance {
  const char* label;
  uint8_t     pin;
  bool        activeLow;
  bool        on;
};

Appliance appliances[] = {
  { "Fan1",  13, true,  false },
  { "Fan2",  23, true,  false },
  { "Fan3",  14, true,  false },
  { "Fan4",  27, true,  false },
  { "Bulb1", 26, true,  false },
  { "Bulb2", 25, true,  false },
  { "Bulb3", 33, true,  false },
  { "Bulb4", 32, true,  false },
  { "Bulb5",  4, false, false },
};
const int NUM_APPLIANCES = sizeof(appliances) / sizeof(appliances[0]);

// Zone n drives appliances[ZONE_APPLIANCE[n]]. Zones are numbered row-major:
//     0 1 2
//     3 4 5
//     6 7 8
//
// Lights and fans alternate across the ceiling in a checkerboard, which is how
// they are physically mounted:
//
//     bulb  fan   bulb
//     fan   bulb  fan
//     bulb  fan   bulb
//
// That pattern needs five lights and four fans, which is exactly the hardware.
// Every zone still gets its own appliance, so an occupied corner lights up
// without switching on the fan next to it.
const int ZONE_APPLIANCE[9] = {
  4, 0, 5,      // Bulb1  Fan1   Bulb2
  1, 6, 2,      // Fan2   Bulb3  Fan3
  7, 3, 8       // Bulb4  Fan4   Bulb5
};

// HC-SR501 motion sensor. GPIO 35 is input-only, which suits a sensor input
// and keeps the output-capable pins free for relays.
const uint8_t PIR_PIN = 35;

// ===========================================================================
// Tuning
// ===========================================================================

// --- turning appliances on and off -----------------------------------------
// ESP-1 sends a fresh grid roughly every 0.7 s, and consecutive frames disagree
// often enough that switching directly on them would chatter the relays. So
// each zone votes: of the last WINDOW frames, how many said "occupied"?
//
// Two thresholds rather than one. Above ON_VOTES the appliance switches on,
// below OFF_VOTES it switches off, and in the gap between them nothing changes.
// That gap is what stops a zone hovering near the boundary from flipping back
// and forth every frame.
const uint8_t  WINDOW      = 20;     // frames remembered per zone (~14 s)
const uint8_t  ON_VOTES    = 12;     // 12/20 = 60% occupied -> switch on
const uint8_t  OFF_VOTES   = 6;      // 6/20  = 30% occupied -> switch off
const uint32_t MIN_HOLD_MS = 4000;   // a zone may not change state faster than this

// --- deciding when ESP-1 should run ----------------------------------------
// Counted in received frames rather than seconds so the threshold is tied to
// how much evidence we have, not to wall-clock time. At ~0.7 s per frame, 150
// frames is a little under two minutes of a completely empty room.
const uint16_t EMPTY_FRAMES_TO_IDLE = 150;

// Any motion blocks the detector from standing down for this long, whatever the
// camera reports. It covers somebody who is in the room but not inside one of
// the nine mapped zones -- at the edge, or in an uncovered corner. The camera
// sees an empty floor and the empty-frame count climbs, but each movement the
// PIR picks up pushes this block forward and the detector keeps running.
const uint32_t MIN_RUN_MS = 60000;

// The HC-SR501 settles for up to a minute after power-up and reports spurious
// motion while it does. Ignore it until then.
const uint32_t PIR_WARMUP_MS = 60000;

// Re-send the current run/idle state this often. ESP-NOW broadcasts are not
// acknowledged, so repeating the state is what makes a lost packet harmless.
const uint32_t CTRL_REPEAT_MS = 2000;

// If the detector is supposed to be running but has said nothing for this long,
// assume the link or the board has failed and switch everything off.
const uint32_t LINK_TIMEOUT_MS = 15000;

// ===========================================================================
// State
// ===========================================================================
const uint8_t BCAST[6] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

// Sliding window of recent raw frames, as a ring buffer plus a running total
// per zone, so adding a frame is O(9) rather than a rescan of the whole window.
uint8_t  history[WINDOW][9];
uint8_t  histHead = 0;
uint8_t  histFill = 0;
uint16_t votes[9];

bool     zoneOccupied[9];      // smoothed verdict, what the relays follow
uint32_t zoneChangedMs[9];

// Whether ESP-1 should be detecting, and the bookkeeping behind that decision.
bool     detectorRunning = true;
uint16_t emptyFrames     = 0;
uint32_t motionHoldUntil = 0;
uint32_t lastCtrlSentMs  = 0;
uint32_t ctrlSeq         = 0;

uint32_t lastFrameMs = 0;
uint32_t lastSeq     = 0;
uint32_t framesRx    = 0;
uint32_t framesLost  = 0;
bool     linkAlive   = false;

int lastPirLevel = LOW;

// Incoming frames are copied out of the WiFi callback into this small ring, so
// the callback stays short and all the real work happens in loop().
const uint8_t QUEUE_LEN = 4;
volatile uint8_t qHead = 0, qTail = 0;
grid_msg_t queue[QUEUE_LEN];

// The receive callback signature changed in version 3 of the ESP32 Arduino
// core. Selecting it with a macro keeps a single function body: the IDE
// generates a prototype for every function it sees without running the
// preprocessor first, so two alternative bodies would produce two prototypes
// and one of them would not compile.
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  #define RECV_ARGS const esp_now_recv_info_t *info, const uint8_t *data, int len
#else
  #define RECV_ARGS const uint8_t *info, const uint8_t *data, int len
#endif

void onGridRecv(RECV_ARGS) {
  (void)info;
  if (len != (int)sizeof(grid_msg_t) || data[0] != MSG_GRID) return;

  uint8_t next = (qHead + 1) % QUEUE_LEN;
  if (next == qTail) return;                  // queue full, drop this frame
  memcpy(&queue[qHead], data, sizeof(grid_msg_t));
  qHead = next;
}

// ===========================================================================
// Relays
// ===========================================================================
void applyAppliance(int i) {
  bool level = appliances[i].on ? appliances[i].activeLow : !appliances[i].activeLow;
  digitalWrite(appliances[i].pin, level ? LOW : HIGH);
}

void setAppliance(int i, bool on) {
  appliances[i].on = on;
  applyAppliance(i);
}

void allAppliancesOff() {
  for (int i = 0; i < NUM_APPLIANCES; i++) setAppliance(i, false);
}

// ===========================================================================
// Sliding window
// ===========================================================================
void clearWindow() {
  memset(history, 0, sizeof(history));
  memset(votes, 0, sizeof(votes));
  histHead = 0;
  histFill = 0;
}

void addFrame(const uint8_t* grid) {
  if (histFill == WINDOW) {
    // Window is full, so the slot about to be overwritten leaves the totals.
    for (int z = 0; z < 9; z++) votes[z] -= history[histHead][z];
  } else {
    histFill++;
  }
  for (int z = 0; z < 9; z++) {
    history[histHead][z] = grid[z] ? 1 : 0;
    votes[z] += history[histHead][z];
  }
  histHead = (histHead + 1) % WINDOW;
}

// Apply the vote thresholds and switch anything that needs switching.
void updateAppliances() {
  for (int z = 0; z < 9; z++) {
    bool want = zoneOccupied[z];                  // between thresholds: hold
    if      (votes[z] >= ON_VOTES)  want = true;
    else if (votes[z] <= OFF_VOTES) want = false;

    if (want == zoneOccupied[z]) continue;
    if (millis() - zoneChangedMs[z] < MIN_HOLD_MS) continue;

    zoneOccupied[z]  = want;
    zoneChangedMs[z] = millis();

    int a = ZONE_APPLIANCE[z];
    setAppliance(a, want);
    Serial.printf("  %s %s  (zone %d, %u of last %u frames occupied)\n",
                  appliances[a].label, want ? "ON" : "OFF", z, votes[z], histFill);
  }
}

bool roomLooksEmpty() {
  for (int z = 0; z < 9; z++) if (zoneOccupied[z]) return false;
  return true;
}

// ===========================================================================
// Control link to ESP-1
// ===========================================================================
void sendControl() {
  ctrl_msg_t msg;
  msg.tag = MSG_CTRL;
  msg.run = detectorRunning ? 1 : 0;
  msg.seq = ++ctrlSeq;
  esp_now_send(BCAST, (const uint8_t*)&msg, sizeof(msg));
  lastCtrlSentMs = millis();
}

void startDetector(const char* why) {
  motionHoldUntil = millis() + MIN_RUN_MS;
  if (detectorRunning) return;

  detectorRunning = true;
  emptyFrames     = 0;
  Serial.printf("\nDetector ON  (%s)\n", why);
  sendControl();
}

void stopDetector(const char* why) {
  if (!detectorRunning) return;

  detectorRunning = false;
  emptyFrames     = 0;
  Serial.printf("\nDetector OFF  (%s)\n", why);

  // Nobody is here, so everything goes off and the window is cleared. Stale
  // votes must not survive into the next occupancy period.
  allAppliancesOff();
  clearWindow();
  for (int z = 0; z < 9; z++) zoneOccupied[z] = false;
  linkAlive = false;

  sendControl();
}

// ===========================================================================
// Reporting
// ===========================================================================
void printStatus() {
  Serial.println(F("--------------------------------"));
  Serial.printf("detector: %s | link: %s | frames: %lu received, %lu lost\n",
                detectorRunning ? "running" : "idle",
                linkAlive ? "up" : "down",
                (unsigned long)framesRx, (unsigned long)framesLost);
  Serial.printf("window: %u/%u frames | empty streak: %u/%u\n",
                histFill, WINDOW, emptyFrames, EMPTY_FRAMES_TO_IDLE);
  for (int z = 0; z < 9; z++) {
    int a = ZONE_APPLIANCE[z];
    Serial.printf("  zone %d  %2u/%-2u votes  %-8s  %-5s %s\n",
                  z, votes[z], histFill,
                  zoneOccupied[z] ? "occupied" : "empty",
                  appliances[a].label,
                  appliances[a].on ? "ON" : "OFF");
  }
  Serial.println(F("--------------------------------"));
}

void printFrame(const grid_msg_t& m) {
  Serial.printf("seq %-5lu raw %d%d%d %d%d%d %d%d%d (%u/9)   verdict %d%d%d %d%d%d %d%d%d (%d/9)\n",
                (unsigned long)m.seq,
                m.grid[0], m.grid[1], m.grid[2],
                m.grid[3], m.grid[4], m.grid[5],
                m.grid[6], m.grid[7], m.grid[8], m.occupied,
                zoneOccupied[0], zoneOccupied[1], zoneOccupied[2],
                zoneOccupied[3], zoneOccupied[4], zoneOccupied[5],
                zoneOccupied[6], zoneOccupied[7], zoneOccupied[8],
                (zoneOccupied[0] + zoneOccupied[1] + zoneOccupied[2] +
                 zoneOccupied[3] + zoneOccupied[4] + zoneOccupied[5] +
                 zoneOccupied[6] + zoneOccupied[7] + zoneOccupied[8]));
}

// ===========================================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  for (int i = 0; i < NUM_APPLIANCES; i++) {
    pinMode(appliances[i].pin, OUTPUT);
    appliances[i].on = false;
    applyAppliance(i);            // everything off before anything else happens
  }
  pinMode(PIR_PIN, INPUT);

  clearWindow();
  for (int z = 0; z < 9; z++) { zoneOccupied[z] = false; zoneChangedMs[z] = 0; }

  Serial.println(F("\n=== Classroom occupancy controller (ESP-2) ==="));

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // The radio starts in the background, and configuring it too early fails
  // quietly. Wait until the interface reports a real MAC address before
  // touching the channel, otherwise ESP-NOW can end up listening on the wrong
  // one and simply never hear anything.
  uint8_t mac[6] = { 0 };
  for (int i = 0; i < 150; i++) {
    if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK &&
        (mac[0] | mac[1] | mac[2] | mac[3] | mac[4] | mac[5]) != 0) break;
    delay(20);
  }

  esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);

  // Power saving lets an idle station switch its receiver off periodically,
  // which for ESP-NOW means quietly missing packets. Turn it off.
  esp_wifi_set_ps(WIFI_PS_NONE);

  uint8_t chan = 0;
  wifi_second_chan_t sec = WIFI_SECOND_CHAN_NONE;
  esp_wifi_get_channel(&chan, &sec);
  Serial.printf("MAC %02X:%02X:%02X:%02X:%02X:%02X, listening on channel %u\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], chan);

  if (esp_now_init() != ESP_OK) {
    Serial.println(F("FATAL: ESP-NOW init failed"));
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(onGridRecv);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BCAST, 6);
  peer.channel = ESPNOW_CHANNEL;
  peer.encrypt = false;
  peer.ifidx   = WIFI_IF_STA;
  esp_now_add_peer(&peer);

  Serial.printf("Switching rule: on at %u/%u frames, off at %u/%u, %lu ms minimum hold\n",
                ON_VOTES, WINDOW, OFF_VOTES, WINDOW, (unsigned long)MIN_HOLD_MS);
  Serial.printf("Detector idles after %u consecutive empty frames; motion restarts it\n",
                EMPTY_FRAMES_TO_IDLE);
  Serial.println(F("Type 's' for status.\n"));

  // Start with the detector running so the room is covered from power-up. The
  // PIR is ignored until it has settled, and if the room really is empty the
  // empty-frame count will idle the detector on its own.
  detectorRunning = true;
  sendControl();
}

void loop() {
  // --- motion ---------------------------------------------------------------
  if (millis() > PIR_WARMUP_MS) {
    int pir = digitalRead(PIR_PIN);
    if (pir == HIGH && lastPirLevel == LOW) {
      startDetector("motion detected");
    }
    lastPirLevel = pir;
  }

  // --- frames from ESP-1 ----------------------------------------------------
  while (qTail != qHead) {
    grid_msg_t m;
    memcpy(&m, &queue[qTail], sizeof(m));
    qTail = (qTail + 1) % QUEUE_LEN;

    if (!linkAlive) {
      Serial.println(F("Link up: receiving from ESP-1"));
      linkAlive = true;
      lastSeq   = 0;
    }
    if (lastSeq && m.seq > lastSeq + 1) framesLost += m.seq - lastSeq - 1;
    lastSeq     = m.seq;
    lastFrameMs = millis();
    framesRx++;

    addFrame(m.grid);
    printFrame(m);
    updateAppliances();

    // Count how long the room has looked empty. This uses the smoothed verdict,
    // not the raw frame, so an isolated false positive does not reset it.
    if (roomLooksEmpty()) emptyFrames++;
    else                  emptyFrames = 0;

    if (emptyFrames >= EMPTY_FRAMES_TO_IDLE && millis() > motionHoldUntil) {
      stopDetector("room empty long enough");
    }
  }

  // --- keep ESP-1 in step ---------------------------------------------------
  if (millis() - lastCtrlSentMs >= CTRL_REPEAT_MS) sendControl();

  // --- the detector should be sending but is not ----------------------------
  if (detectorRunning && linkAlive && millis() - lastFrameMs > LINK_TIMEOUT_MS) {
    Serial.println(F("\nLink lost: nothing from ESP-1, switching everything off"));
    linkAlive = false;
    allAppliancesOff();
    clearWindow();
    for (int z = 0; z < 9; z++) zoneOccupied[z] = false;
  }

  // --- console --------------------------------------------------------------
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("s")) printStatus();
  }

  delay(5);
}
