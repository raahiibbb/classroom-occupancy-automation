/*
  ESP-2 -- OCCUPANCY -> FAN/BULB CONTROLLER  (classic ESP32 "ESP-32S 30P" NodeMCU)
  ============================================================================
  Receives the 3x3 occupancy grid broadcast over ESP-NOW by ESP-1 (the ESP32-S3
  + OV3660 camera board running the detector), smooths it over a sliding window,
  and drives 9 relays -- 4 fans + 5 bulbs -- one per zone. It also decides when
  ESP-1 should run at all (PIR wake-up, stand-down when the room is empty) and
  tells it so over the same two-way ESP-NOW link.

  FRESH START: powering up or resetting EITHER board returns the whole system
  to idle -- room assumed empty, ESP-1 asleep (no grids), appliances off, LED
  off. Only PIR motion wakes ESP-1. If only ESP-1 reboots, it sends HELLO and
  this board resets itself to match.

  WHY THE SMOOTHING: ESP-1 ships its RAW per-frame model output, which flickers
  (a cell can flip 0/1 between consecutive frames even on a static scene).
  Switching a relay on every flip would chatter the contacts. So ESP-2 keeps the
  last WINDOW frames per zone and applies a vote + hysteresis + dwell rule --
  see THRESHOLD RULE below.

  ARDUINO IDE SETUP
    occupancy_link.h MUST sit in this same folder -- the IDE only finds headers
    next to the sketch. It must be identical to ESP-1's esp1_detector/include/ copy.
    Tools -> Board : "ESP32 Dev Module"  (or "NodeMCU-32S")
    Tools -> Port  : the port that appears when the NodeMCU is plugged in
    Upload, then open Serial Monitor @ 115200 baud, line ending = Newline.
    If upload stalls at "Connecting....", hold BOOT, tap EN/RST, release BOOT.

  WIRING -- relays (pins and polarity unchanged; only the zone mapping moved to
  a CHECKERBOARD matching the ceiling):
    Zone 0 -> Bulb 1 -> GPIO 26        Zone 5 -> Fan 3  -> GPIO 14
    Zone 1 -> Fan 1  -> GPIO 13        Zone 6 -> Bulb 4 -> GPIO 32
    Zone 2 -> Bulb 2 -> GPIO 25        Zone 7 -> Fan 4  -> GPIO 27
    Zone 3 -> Fan 2  -> GPIO 23  (*)   Zone 8 -> Bulb 5 -> GPIO 4  (active-HIGH)
    Zone 4 -> Bulb 3 -> GPIO 33
    Relay board VCC -> its OWN 5 V supply; GND common with the ESP32.
    Fans 1-4 and Bulbs 1-4 are active-LOW relays; Bulb 5 is the one active-HIGH.

  WIRING -- sensor and indicator:
    HC-SR501 PIR : VCC -> 5V,  GND -> GND,  OUT -> GPIO 35 (input-only pin)
    Status LED   : GPIO 19 -> 220-330 ohm resistor -> LED anode, cathode -> GND.
                   ON only while ESP-1 is awake AND proving it by sending grid
                   frames; ESP-2's run command alone never lights it. OFF at
                   power-up/reset (fresh start), immediately at stand-down,
                   and within 3 s if ESP-1 goes quiet while it should be running.

  ZONE NUMBERING: the grid is row-major, so zone = row*3 + col --
        Z0 Z1 Z2          Bulb1  Fan1   Bulb2
        Z3 Z4 Z5    ->    Fan2   Bulb3  Fan3
        Z6 Z7 Z8          Bulb4  Fan4   Bulb5
    Re-map which zone drives which appliance by reordering ZONE_APPLIANCE[].

  (*) FAN 2 MOVED OFF GPIO 12 -- REWIRE REQUIRED.
  GPIO 12 is the classic ESP32's MTDI boot strapping pin: its level at reset
  selects the flash regulator voltage. The relay board holds it HIGH, which
  selects 1.8 V while the flash chip is 3.3 V. Confirmed on this board with
  `esptool flash-id`, which reported:
      Manufacturer: ff / Device: ffff / Detected flash size: Unknown
      Flash voltage set by a strapping pin: 1.8V
  i.e. the flash is unreadable. That breaks BOTH uploading and booting, and it
  is read on every reset -- so it cannot be worked around in software.
  Fan 2 is therefore on GPIO 23 (no strapping function, safe as an output).
  >>> Physically move the Fan 2 relay IN wire from pin D12 to pin D23. <<<
  Leave GPIO 12 unconnected.

  Other pins that are NOT safe here, for reference: GPIO 0, 2, 5, 15 (all
  strapping), and 6-11 (wired to the SPI flash). Free alternatives if GPIO 23
  is inconvenient: 18, 17, 16 (19 is now the status LED, 35 the PIR).

  SERIAL COMMANDS (type + Enter in the Serial Monitor):
    s -> print status now
    v -> toggle full per-frame block vs ONE compact line per frame (easier to
         read; still shows PIR + LED state on every line)
*/

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ===========================================================================
// THRESHOLD RULE -- how the flickering per-frame output becomes a verdict
// ===========================================================================
// For each zone independently, ESP-2 keeps the last WINDOW raw frames and counts
// how many of them said "occupied" (that count is the zone's "votes"). Then,
// with hysteresis -- two thresholds, not one:
//
//   votes >= ON_VOTES   -> zone is OCCUPIED  -> appliance ON
//   votes <= OFF_VOTES  -> zone is EMPTY     -> appliance OFF
//   in between          -> DEAD BAND: hold whatever the zone already was
//
// The dead band is what stops a zone sitting near the line from buzzing the
// relay on and off. On top of that a zone may not change state more often than
// MIN_HOLD_MS, which caps the switching rate even in a worst case.
//
// ESP-1 sends ~1 frame every 0.7 s, so WINDOW=20 is a ~14 s memory: switching
// something ON needs 12 of the last 20 frames occupied (60%), and switching it
// OFF needs the zone to fall back to 6 or fewer (30%).
static const uint8_t  WINDOW      = 20;   // frames of history per zone
static const uint8_t  ON_VOTES    = 12;   // >= this many occupied -> switch ON
static const uint8_t  OFF_VOTES   = 6;    // <= this many occupied -> switch OFF
static const uint32_t MIN_HOLD_MS = 4000; // min time between flips of one zone

// If ESP-1 goes quiet this long WHILE IT IS SUPPOSED TO BE RUNNING, assume the
// link or camera died and fail safe: everything OFF, history cleared. Silence
// while ESP-1 is idle is expected and must not trip this.
static const uint32_t LINK_TIMEOUT_MS = 15000;

// ===========================================================================
// Wire format -- grid_msg_t / ctrl_msg_t / MSG_GRID / MSG_CTRL / ESPNOW_CHANNEL
// all come from occupancy_link.h, which must sit in THIS sketch folder (the
// Arduino IDE only finds headers beside the .ino) and must stay byte-for-byte
// identical to ESP-1's copy in esp1_detector/include/.
// ===========================================================================
#include "occupancy_link.h"

static const uint8_t BCAST[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ===========================================================================
// RUN / IDLE POLICY -- when ESP-1 should be detecting at all
// ===========================================================================
// ESP-2 owns this decision: the PIR is wired here, and so is the smoothed,
// flicker-free view of the room. ESP-1 just obeys.
//
//   WAKE:  any PIR motion (after the sensor's warm-up) -> tell ESP-1 to run.
//   IDLE:  EMPTY_FRAMES_TO_IDLE consecutive received frames in which the
//          SMOOTHED verdict has every zone empty -> tell ESP-1 to idle, switch
//          everything off, clear the voting window.
//   HOLD:  any motion blocks a stand-down for MOTION_HOLD_MS afterwards. This
//          covers someone in the room but outside all nine zones: the camera
//          sees an empty floor, but the PIR overrules it.
static const uint8_t  PIR_PIN              = 35;     // input-only pin; HC-SR501 OUT
// The HC-SR501 can false-trigger for up to a minute after it is powered. Cut
// from 60 s to 10 s at the user's request: a false trigger here can only wake
// ESP-1 for ~2 min in an empty room (LED on, no appliance switches, because the
// relays still follow the camera's 12-of-20 vote), so a short warm-up is a
// cheap trade. If "MOTION DETECTED" banners appear right after boot with nobody
// around, raise this to 20000-30000.
static const uint32_t PIR_WARMUP_MS        = 10000;
static const uint16_t EMPTY_FRAMES_TO_IDLE = 150;    // ~1.8 min at ~0.72 s/frame
static const uint32_t MOTION_HOLD_MS       = 60000;  // motion blocks stand-down this long

// Run/idle is re-broadcast as REPEATED STATE, not a one-shot command. ESP-NOW
// broadcasts are unacknowledged, so a single lost "idle" packet would otherwise
// leave ESP-1 running forever; repeating means the next packet corrects it.
static const uint32_t CTRL_REPEAT_MS = 2000;

// Status LED: shows whether ESP-1 is ACTUALLY AWAKE, as reported by ESP-1
// itself -- not what ESP-2 has asked for. ESP-1 only transmits a grid frame
// after it has captured an image and run the model, so every frame is ESP-1
// signalling "I am awake and working". That is stronger evidence than a bare
// "I'm awake" flag, which would only prove the chip has power.
//   ON  : a grid frame arrived within the last ESP1_AWAKE_TIMEOUT_MS
//   OFF : ESP-1 has gone quiet -- idled on command, or dead/unpowered
// Wire GPIO 19 -> 220-330 ohm resistor -> LED anode, LED cathode -> GND.
// Active-HIGH. GPIO 19 has no strapping role, so it is safe at boot.
static const uint8_t STATUS_LED_PIN = 19;

// ESP-1 sends a frame every ~0.72 s, so 3 s is ~4 frames: long enough that one
// or two lost packets do not blink the LED off, short enough to react promptly.
static const uint32_t ESP1_AWAKE_TIMEOUT_MS = 3000;

// ===========================================================================
// Appliance table (identical wiring/polarity to the standalone test sketch)
// ===========================================================================
struct Appliance {
  const char* label;
  int  pin;
  bool activeLow;   // true = active-LOW relay (LOW = ON), false = active-HIGH
  bool state;       // current logical state (true = ON)
};

Appliance appliances[] = {
  {"Fan1",  13, true,  false},
  {"Fan2",  23, true,  false},   // was GPIO 12 -- MTDI strapping pin, see header
  {"Fan3",  14, true,  false},
  {"Fan4",  27, true,  false},
  {"Bulb1", 26, true,  false},
  {"Bulb2", 25, true,  false},
  {"Bulb3", 33, true,  false},
  {"Bulb4", 32, true,  false},
  {"Bulb5",  4, false, false},   // the single active-HIGH relay
};
const int NUM_APPLIANCES = sizeof(appliances) / sizeof(appliances[0]);

// Zone -> appliance index, as a CHECKERBOARD matching the ceiling layout.
// Indices refer to appliances[] above (0-3 = Fan1-4, 4-8 = Bulb1-5):
//
//        Bulb1  Fan1   Bulb2          4  0  5
//        Fan2   Bulb3  Fan3     =     1  6  2
//        Bulb4  Fan4   Bulb5          7  3  8
//
// Pins and relay polarity are unchanged -- only which zone drives which relay.
const int ZONE_APPLIANCE[9] = { 4, 0, 5,
                                1, 6, 2,
                                7, 3, 8 };

// ===========================================================================
// Sliding-window state
// ===========================================================================
uint8_t  hist[WINDOW][9];   // ring buffer of the last WINDOW raw frames
uint8_t  histHead = 0;      // next slot to write
uint8_t  histFill = 0;      // how many slots are valid (saturates at WINDOW)
uint16_t votes[9];          // running per-zone sum over the ring
bool     verdict[9];        // smoothed occupancy actually being acted on
uint32_t lastChangeMs[9];   // for MIN_HOLD_MS

// ===========================================================================
// ESP-NOW receive -> small ring queue. The callback runs in the WiFi task, so
// it only enqueues; all printing and relay work happens in loop().
// ===========================================================================
static const uint8_t QDEPTH = 4;
volatile uint8_t qHead = 0, qTail = 0;
grid_msg_t qBuf[QDEPTH];
uint8_t    qSrc[QDEPTH][6];

uint32_t lastRxMs   = 0;
uint32_t lastSeq    = 0;
uint32_t totalDrops = 0;
uint32_t totalRx    = 0;
bool     linkAlive  = false;
bool     verbose    = true;

// --- run / idle state (see RUN / IDLE POLICY above) -------------------------
// FRESH START: every power-up or reset begins IDLE -- room assumed empty, ESP-1
// told to idle, all appliances and the LED off. Only PIR motion wakes ESP-1.
bool     detectorRunning = false;

// Set by the receive callback when ESP-1 announces it has just rebooted
// (MSG_HELLO); handled in loop() by resetting the whole system to fresh idle.
volatile bool     helloPending = false;
volatile uint32_t hellosRx     = 0;
uint16_t emptyFrames     = 0;      // consecutive frames with every zone empty
bool     motionSeen      = false;  // any PIR motion since the last fresh start?
uint32_t lastMotionMs    = 0;      // millis() of the most recent PIR motion
uint32_t lastCtrlSentMs  = 0;
uint32_t ctrlSeq         = 0;
int      lastPirLevel    = LOW;    // last raw PIR OUT level, for edge logging
bool     pirArmed        = false;  // becomes true once PIR_WARMUP_MS has passed
bool     lastHoldActive  = false;  // to announce when the motion hold expires

// --- status LED: is ESP-1 actually awake? ------------------------------------
// Lit only by grid frames arriving from ESP-1 while it is commanded to run;
// ESP-2's command alone never lights it. Starts false (fresh start = LED off).
bool     esp1Awake       = false;
uint32_t lastGridAnyMs   = 0;      // last valid grid frame accepted while running

// True while recent PIR motion is blocking a stand-down.
// Written as "elapsed since last motion < hold" with UNSIGNED subtraction, which
// stays correct across a millis() wrap (~49.7 days) for any interval -- a
// classroom controller can easily run that long. The motionSeen flag makes
// "no motion yet" unambiguous (an earlier signed-difference version read a
// never-set hold as ACTIVE once uptime passed ~24.8 days).
bool motionHoldActive() {
  return motionSeen && (millis() - lastMotionMs) < MOTION_HOLD_MS;
}

// Whole seconds of motion hold remaining (0 when no hold).
uint32_t motionHoldLeftS() {
  if (!motionHoldActive()) return 0;
  return (MOTION_HOLD_MS - (millis() - lastMotionMs) + 999) / 1000;
}

// Whole seconds left in the PIR warm-up (0 once armed). Guarded so it can never
// underflow into a huge number in the moment millis() crosses the boundary.
uint32_t pirWarmupLeftS() {
  uint32_t now = millis();
  return (now < PIR_WARMUP_MS) ? (PIR_WARMUP_MS - now + 999) / 1000 : 0;
}

// One self-contained line describing the PIR right now. Printed at the top of
// every frame block, in the idle heartbeat and in the status report, so the
// current PIR state is always on the most recent lines of the monitor instead
// of scrolling away under the ~0.7 s frame output.
void printPirLine() {
  int lvl = digitalRead(PIR_PIN);
  if (!pirArmed) {
    Serial.printf("PIR : WARMING UP, %lu s left (motion ignored until armed) | OUT=%s\n",
                  (unsigned long)pirWarmupLeftS(), lvl == HIGH ? "HIGH" : "LOW");
    return;
  }
  Serial.printf("PIR : ARMED | OUT=%s | stand-down hold: ",
                lvl == HIGH ? "HIGH (motion now)" : "LOW (no motion)");
  if (motionHoldActive())
    Serial.printf("%lu s left\n", (unsigned long)motionHoldLeftS());
  else
    Serial.println(F("none"));
}

// One line for the LED and what it reflects.
void printLedLine() {
  Serial.printf("LED : %s -- ESP-1 %s | ESP-2 is commanding ESP-1 to %s\n",
                esp1Awake ? "ON " : "OFF",
                esp1Awake ? "AWAKE (frames arriving)" : "NOT sending frames",
                detectorRunning ? "RUN" : "IDLE");
}

// Raw callback counters. These count EVERY callback the radio delivers, before
// any validation -- so "nothing is arriving" can be told apart from "frames are
// arriving but being rejected", which the packet counter alone cannot show.
volatile uint32_t rawCallbacks = 0;
volatile uint32_t badLenCount  = 0;
volatile int      lastBadLen   = 0;

static void enqueue(const uint8_t* mac, const uint8_t* data, int len) {
  rawCallbacks = rawCallbacks + 1;   // plain ++ on a volatile is deprecated in C++20

  // ESP-1 has just rebooted: flag it, and loop() resets the system to fresh idle.
  if (len == (int)sizeof(hello_msg_t) && data[0] == MSG_HELLO) {
    helloPending = true;
    hellosRx = hellosRx + 1;
    return;
  }

  // Otherwise accept only a grid frame: right length AND right tag. Checking
  // the tag too means a stray ESP-NOW packet of the same size from some other
  // device on this channel cannot be mistaken for ESP-1's output.
  if (len != (int)sizeof(grid_msg_t) || data[0] != MSG_GRID) {
    badLenCount = badLenCount + 1;
    lastBadLen = len;
    return;
  }
  uint8_t next = (uint8_t)((qHead + 1) % QDEPTH);
  if (next == qTail) return;                 // queue full -- drop this arrival
  memcpy(&qBuf[qHead], data, sizeof(grid_msg_t));
  memcpy(qSrc[qHead], mac, 6);
  qHead = next;
}

// Arduino-ESP32 core 3.x changed this callback's signature (it now passes an
// esp_now_recv_info_t instead of a bare MAC). Select it with a macro rather
// than #if-ing two whole function bodies: the Arduino IDE auto-generates a
// prototype for every function definition it sees WITHOUT preprocessing, so two
// bodies would yield two prototypes -- and on core 2.x the 3.x one would fail
// to compile. One body + a macro leaves exactly one prototype to expand.
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  #define RECV_SIG const esp_now_recv_info_t *info, const uint8_t *data, int len
  #define RECV_MAC info->src_addr
#else
  #define RECV_SIG const uint8_t *mac, const uint8_t *data, int len
  #define RECV_MAC mac
#endif

void onRecv(RECV_SIG) {
  enqueue(RECV_MAC, data, len);
}

// ===========================================================================
// Relay helpers
// ===========================================================================
void applyState(int i) {
  bool on = appliances[i].state, activeLow = appliances[i].activeLow;
  digitalWrite(appliances[i].pin, on ? (activeLow ? LOW : HIGH)
                                     : (activeLow ? HIGH : LOW));
}

void setAppliance(int i, bool on) {
  appliances[i].state = on;
  applyState(i);
}

void allOff() {
  for (int i = 0; i < NUM_APPLIANCES; i++) setAppliance(i, false);
}

// ===========================================================================
// Window bookkeeping
// ===========================================================================
void resetWindow() {
  memset(hist, 0, sizeof(hist));
  memset(votes, 0, sizeof(votes));
  histHead = 0;
  histFill = 0;
}

void pushFrame(const uint8_t* grid) {
  if (histFill == WINDOW) {                       // ring full -- evict oldest
    for (int z = 0; z < 9; z++) votes[z] -= hist[histHead][z];
  } else {
    histFill++;
  }
  for (int z = 0; z < 9; z++) {
    hist[histHead][z] = grid[z] ? 1 : 0;
    votes[z] += hist[histHead][z];
  }
  histHead = (uint8_t)((histHead + 1) % WINDOW);
}

// ===========================================================================
// Printing
// ===========================================================================
void printStatus() {
  Serial.println(F("---------------- STATUS ----------------"));
  printPirLine();
  printLedLine();
  Serial.printf("ctrl msgs sent: %lu | ESP-1 reboots seen (HELLO): %lu | empty streak: %u/%u frames\n",
                (unsigned long)ctrlSeq, (unsigned long)hellosRx, emptyFrames, EMPTY_FRAMES_TO_IDLE);
  Serial.printf("link: %s | frames rx: %lu | dropped: %lu | window: %u/%u\n",
                linkAlive ? "UP" : "DOWN", (unsigned long)totalRx,
                (unsigned long)totalDrops, histFill, WINDOW);
  for (int z = 0; z < 9; z++) {
    int a = ZONE_APPLIANCE[z];
    Serial.printf("  Z%d %2u/%-2u votes -> %-8s  %-6s = %s\n",
                  z, votes[z], histFill,
                  verdict[z] ? "OCCUPIED" : "empty",
                  appliances[a].label,
                  appliances[a].state ? "ON" : "OFF");
  }
  Serial.println(F("----------------------------------------"));
}

void printFrame(const grid_msg_t& m, const uint8_t* src) {
  Serial.println();
  Serial.printf("=== seq %lu  from %02X:%02X:%02X:%02X:%02X:%02X  infer %u ms  drops %lu ===\n",
                (unsigned long)m.seq, src[0], src[1], src[2], src[3], src[4], src[5],
                m.infer_ms, (unsigned long)totalDrops);

  // --- PIR + LED first, so they are always in the newest block on screen ---
  printPirLine();
  printLedLine();

  // --- what ESP-1 actually said this frame (raw, unsmoothed) ---
  Serial.printf("ESP-1 RAW grid (this frame): %u/9 occupied\n", m.occupied);
  for (int r = 0; r < 3; r++) {
    Serial.print(F("      "));
    for (int c = 0; c < 3; c++) {
      Serial.print(m.grid[r * 3 + c]);
      if (c != 2) Serial.print(F("  "));
    }
    Serial.println();
  }

  // --- the vote tally the verdict is based on ---
  Serial.printf("Votes over last %u frames (ON if >=%u, OFF if <=%u):\n",
                histFill, ON_VOTES, OFF_VOTES);
  for (int r = 0; r < 3; r++) {
    Serial.print(F("      "));
    for (int c = 0; c < 3; c++) {
      int z = r * 3 + c;
      Serial.printf("Z%d %2u/%-2u   ", z, votes[z], histFill);
    }
    Serial.println();
  }

  // --- the smoothed verdict ---
  int n = 0;
  for (int z = 0; z < 9; z++) if (verdict[z]) n++;
  Serial.printf("FINAL VERDICT (smoothed): %d/9 occupied\n", n);
  for (int r = 0; r < 3; r++) {
    Serial.print(F("      "));
    for (int c = 0; c < 3; c++) Serial.printf("%-6s", verdict[r * 3 + c] ? "OCC" : "-");
    Serial.println();
  }

  // --- progress toward a stand-down ---
  Serial.printf("Stand-down: empty streak %u/%u frames%s\n",
                emptyFrames, EMPTY_FRAMES_TO_IDLE,
                motionHoldActive() ? "  [PIR motion hold active]" : "");

  // --- what is physically being driven right now, laid out like the room ---
  // Printed as the same 3x3 grid as the camera view, so the checkerboard
  // (bulb/fan alternating) can be checked against the ceiling at a glance.
  Serial.println(F("DRIVING NOW (room layout):"));
  for (int r = 0; r < 3; r++) {
    Serial.print(F("      "));
    for (int c = 0; c < 3; c++) {
      int a = ZONE_APPLIANCE[r * 3 + c];
      Serial.printf("%-5s=%-4s  ", appliances[a].label, appliances[a].state ? "ON" : "OFF");
    }
    Serial.println();
  }
}

// Quiet mode ('v'): ONE line per frame instead of the full block, with the PIR
// and LED state on it -- far easier to follow at ~1.4 frames/s.
void printCompact(const grid_msg_t& m) {
  int n = 0;
  for (int z = 0; z < 9; z++) if (verdict[z]) n++;
  Serial.printf("seq %-6lu raw %u/9 | occupied %d/9 | empty %3u/%u | ",
                (unsigned long)m.seq, m.occupied, n, emptyFrames, EMPTY_FRAMES_TO_IDLE);
  if (!pirArmed) {
    Serial.printf("PIR warm-up %2lus OUT=%s", (unsigned long)pirWarmupLeftS(),
                  digitalRead(PIR_PIN) == HIGH ? "HI" : "lo");
  } else {
    Serial.printf("PIR OUT=%s hold=", digitalRead(PIR_PIN) == HIGH ? "HI" : "lo");
    if (motionHoldActive()) Serial.printf("%2lus", (unsigned long)motionHoldLeftS());
    else                    Serial.print(F("none"));
  }
  Serial.printf(" | LED %s\n", esp1Awake ? "ON" : "OFF");
}

// ===========================================================================
// Run / idle control of ESP-1
// ===========================================================================
// The LED reflects ESP-1's REPORTED state (esp1Awake, driven by its frames),
// not the run/idle command we send. startDetector()/stopDetector() below do
// NOT touch it -- only frames arriving, or ESP-1 falling silent, do.
void setStatusLed() {
  digitalWrite(STATUS_LED_PIN, esp1Awake ? HIGH : LOW);
}

// Smoothed verdict, NOT the raw frame: a single false-positive frame cannot
// reset the empty count, and a single false-negative cannot advance it.
bool roomLooksEmpty() {
  for (int z = 0; z < 9; z++) if (verdict[z]) return false;
  return true;
}

// Broadcast the CURRENT desired state. Called on every change and then every
// CTRL_REPEAT_MS regardless, so a lost packet is corrected by the next one.
void sendControl() {
  ctrl_msg_t msg;
  msg.tag = MSG_CTRL;
  msg.run = detectorRunning ? 1 : 0;
  msg.seq = ++ctrlSeq;
  esp_now_send(BCAST, (const uint8_t*)&msg, sizeof(msg));
  lastCtrlSentMs = millis();
}

void clearVerdicts() {
  resetWindow();
  for (int z = 0; z < 9; z++) verdict[z] = false;
}

void startDetector(const char* why) {
  // Motion always pushes the stand-down block forward, even when ESP-1 is
  // already running -- that is how the PIR overrules an empty-looking floor.
  motionSeen   = true;
  lastMotionMs = millis();
  if (detectorRunning) return;

  detectorRunning = true;
  emptyFrames     = 0;
  Serial.printf("\n>>> DETECTOR ON  (%s) -- telling ESP-1 to RUN "
                "(LED lights once ESP-1's frames arrive)\n", why);
  sendControl();
}

void stopDetector(const char* why) {
  if (!detectorRunning) return;

  detectorRunning = false;
  emptyFrames     = 0;
  Serial.printf("\n>>> DETECTOR OFF (%s) -- telling ESP-1 to SLEEP, all appliances OFF, "
                "window cleared, LED OFF\n", why);

  // Nobody is here: everything off, and clear the window so stale votes from
  // this occupancy period cannot leak into the next one.
  allOff();
  clearVerdicts();
  linkAlive = false;   // ESP-1 falling silent now is EXPECTED, not a fault

  // LED off now. Frames are ignored while idle (see the queue drain), so the
  // last in-flight frame or two from ESP-1 cannot re-light it.
  esp1Awake = false;
  setStatusLed();

  sendControl();
}

// FRESH START: back to the exact state of a clean power-up -- room assumed
// empty, ESP-1 idle, all appliances off, window cleared, LED off, no motion
// hold. Used when ESP-1 announces it has rebooted (HELLO), so that resetting
// EITHER board resets the whole system. Safe to repeat (ESP-1 may send a few
// HELLOs before it hears our IDLE). The PIR's armed state is kept: the sensor
// is powered from this board, so ESP-1 rebooting does not disturb it.
void freshStart(const char* why) {
  detectorRunning = false;
  emptyFrames     = 0;
  motionSeen      = false;          // no hold carried over from before
  lastHoldActive  = false;
  allOff();
  clearVerdicts();
  linkAlive = false;
  esp1Awake = false;
  setStatusLed();
  sendControl();                    // IDLE -- this is also ESP-1's boot acknowledgement
  Serial.printf("\n>>> FRESH START (%s): ESP-1 told to SLEEP, all appliances OFF, "
                "LED OFF -- waiting for PIR motion\n", why);
}

// ===========================================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  for (int i = 0; i < NUM_APPLIANCES; i++) {
    pinMode(appliances[i].pin, OUTPUT);
    appliances[i].state = false;
    applyState(i);                 // everything OFF at boot, whatever the polarity
  }
  resetWindow();
  for (int z = 0; z < 9; z++) { verdict[z] = false; lastChangeMs[z] = 0; }

  // Status LED starts OFF (esp1Awake = false). It lights only when ESP-1's
  // first frame arrives and proves ESP-1 is actually up and detecting.
  pinMode(STATUS_LED_PIN, OUTPUT);
  setStatusLed();

  // GPIO 35 is input-only with no internal pull resistors; the HC-SR501 drives
  // its OUT pin actively (about 3.3 V high), so a plain INPUT is correct.
  pinMode(PIR_PIN, INPUT);

  Serial.println();
  Serial.println(F("--- ESP-2 OCCUPANCY -> FAN/BULB CONTROLLER (ESP-NOW) ---"));

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // Arduino-ESP32 core 3.x brings the radio up ASYNCHRONOUSLY. Any esp_wifi_*
  // call made before it is actually running fails silently, and WiFi.macAddress()
  // comes back as 00:00:00:00:00:00 -- which also means the channel never got
  // set, so ESP-NOW would listen on the wrong channel and hear nothing.
  // So: wait for a real MAC first, then configure, then verify.
  uint8_t mac[6] = {0};
  bool wifiUp = false;
  for (int i = 0; i < 150; i++) {                   // up to ~3 s
    if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK &&
        (mac[0] | mac[1] | mac[2] | mac[3] | mac[4] | mac[5]) != 0) { wifiUp = true; break; }
    delay(20);
  }
  Serial.printf("Receiver MAC: %02X:%02X:%02X:%02X:%02X:%02X%s\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
                wifiUp ? "" : "   <-- ALL ZEROS");
  if (!wifiUp)
    Serial.println(F("*** WARNING: WiFi radio did not start - ESP-NOW cannot receive ***"));

  esp_err_t ce = esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  uint8_t prim = 0; wifi_second_chan_t sec = WIFI_SECOND_CHAN_NONE;
  esp_wifi_get_channel(&prim, &sec);
  Serial.printf("set_channel(%d) -> %s | radio is actually on channel %u\n",
                ESPNOW_CHANNEL, esp_err_to_name(ce), prim);
  if (prim != ESPNOW_CHANNEL)
    Serial.println(F("*** WARNING: channel mismatch - ESP-1 and ESP-2 must agree ***"));

  // CRITICAL for an ESP-NOW receiver: Arduino enables WiFi power save by
  // default in STA mode, so an unassociated station duty-cycles its radio and
  // silently misses most/all incoming ESP-NOW frames. Turn it off.
  esp_err_t pe = esp_wifi_set_ps(WIFI_PS_NONE);
  Serial.printf("esp_wifi_set_ps(NONE) -> %s\n", esp_err_to_name(pe));

  esp_err_t ne = esp_now_init();
  Serial.printf("esp_now_init() -> %s\n", esp_err_to_name(ne));
  if (ne != ESP_OK) {
    Serial.println(F("FATAL: esp_now_init failed"));
    while (1) delay(1000);
  }
  esp_err_t re = esp_now_register_recv_cb(onRecv);
  Serial.printf("esp_now_register_recv_cb() -> %s\n", esp_err_to_name(re));

  // The link is now TWO-WAY: ESP-2 sends run/idle to ESP-1, so it needs a
  // broadcast peer just like ESP-1 has. Without this, esp_now_send() fails and
  // ESP-1 would never hear a stand-down.
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BCAST, 6);
  peer.channel = ESPNOW_CHANNEL;
  peer.encrypt = false;
  peer.ifidx   = WIFI_IF_STA;
  esp_err_t ae = esp_now_add_peer(&peer);
  Serial.printf("esp_now_add_peer(broadcast) -> %s\n", esp_err_to_name(ae));

  Serial.printf("sizeof(grid_msg_t) = %u, ctrl_msg_t = %u, hello_msg_t = %u bytes (expect 17/6/5)\n",
                (unsigned)sizeof(grid_msg_t), (unsigned)sizeof(ctrl_msg_t),
                (unsigned)sizeof(hello_msg_t));

  Serial.printf("Listening on channel %d. Rule: ON if >=%u of last %u frames occupied,\n",
                ESPNOW_CHANNEL, ON_VOTES, WINDOW);
  Serial.printf("OFF if <=%u; in between hold; min %lu ms between flips.\n",
                OFF_VOTES, (unsigned long)MIN_HOLD_MS);
  Serial.printf("Stand-down after %u consecutive all-empty frames; PIR on GPIO %u "
                "armed after %lu s warm-up; motion blocks stand-down for %lu s.\n",
                EMPTY_FRAMES_TO_IDLE, PIR_PIN,
                (unsigned long)(PIR_WARMUP_MS / 1000), (unsigned long)(MOTION_HOLD_MS / 1000));
  Serial.println(F("Commands: s = status, v = full block / one line per frame"));
  Serial.println(F("\nFRESH START: room assumed EMPTY, ESP-1 told to SLEEP, all appliances OFF, LED OFF."));
  Serial.println(F("Only PIR motion wakes ESP-1."));
  Serial.printf("PIR warming up: motion is ignored for the first %lu s.\n\n",
                (unsigned long)(PIR_WARMUP_MS / 1000));

  // Announce the initial IDLE straight away rather than waiting for the first
  // repeat tick. If ESP-1 was running before this reset, it stops now; if ESP-1
  // has just booted too, this IDLE is also the acknowledgement it waits for.
  sendControl();
}

void loop() {
  // ---- serial commands ----
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("s")) {
      printStatus();
    } else if (cmd.equalsIgnoreCase("v")) {
      verbose = !verbose;
      Serial.printf("verbose = %s\n", verbose ? "ON" : "OFF");
    } else if (cmd.length()) {
      Serial.print(F("Unknown command: "));
      Serial.println(cmd);
    }
  }

  // ---- ESP-1 rebooted? -> reset the whole system to fresh idle ----
  // Handled before the PIR on purpose: if someone is moving right now, the PIR
  // check below then wakes ESP-1 straight back up, which is the right outcome.
  if (helloPending) {
    helloPending = false;
    freshStart("ESP-1 rebooted");
  }

  // ---- PIR motion ----
  // Ignored for PIR_WARMUP_MS after boot: the HC-SR501 false-triggers while it
  // settles. After that, motion wakes ESP-1 and holds off any stand-down.
  //
  // Deliberately LEVEL-triggered for the hold, not edge-only: in its default
  // repeat-trigger mode the HC-SR501 keeps OUT high for as long as motion
  // continues. Refreshing only on the rising edge would let the hold expire 60 s
  // after motion STARTED and stand down on someone who is still moving. The
  // rising edge is used only for the log line, so the monitor is not spammed.
  //
  // The pin is read on EVERY pass, even during warm-up, so every PIR event is
  // logged -- you can check the wiring immediately, without waiting for arming.
  // Events print as boxed banners so they stand out from the frame output.
  if (!pirArmed && millis() >= PIR_WARMUP_MS) {
    pirArmed = true;
    Serial.println(F("\n##################################################"));
    Serial.printf("##  PIR ARMED -- %2lu s warm-up over.             ##\n",
                  (unsigned long)(PIR_WARMUP_MS / 1000));
    Serial.println(F("##  Motion now wakes the detector.              ##"));
    Serial.println(F("##################################################\n"));
  }

  int pir = digitalRead(PIR_PIN);
  if (pir != lastPirLevel) {
    if (pir == HIGH) {
      if (pirArmed) {
        Serial.println(F("\n##################################################"));
        Serial.println(F("##  PIR: MOTION DETECTED  (OUT went HIGH)       ##"));
        Serial.println(F("##################################################"));
      } else {
        Serial.printf("\n## PIR: motion seen during warm-up -- IGNORED (%lu s until armed). "
                      "Wiring works. ##\n", (unsigned long)pirWarmupLeftS());
      }
    } else {
      if (pirArmed)
        Serial.printf("\n## PIR: motion ended (OUT went LOW). %lu s stand-down hold now counting down. ##\n",
                      (unsigned long)(MOTION_HOLD_MS / 1000));
      else
        Serial.println(F("\n## PIR: OUT went LOW (still warming up) ##"));
    }
    lastPirLevel = pir;
  }
  if (pirArmed && pir == HIGH) {
    startDetector("PIR motion");     // no-op if already running, but refreshes the hold
  }

  // Announce the moment the motion hold runs out, so it is clear when a
  // stand-down becomes possible again.
  bool holdNow = motionHoldActive();
  if (lastHoldActive && !holdNow)
    Serial.printf("\n## PIR: %lu s motion hold EXPIRED -- stand-down allowed again ##\n",
                  (unsigned long)(MOTION_HOLD_MS / 1000));
  lastHoldActive = holdNow;

  // ---- drain the ESP-NOW queue ----
  while (qTail != qHead) {
    grid_msg_t m;
    uint8_t src[6];
    memcpy(&m, &qBuf[qTail], sizeof(m));
    memcpy(src, qSrc[qTail], 6);
    qTail = (uint8_t)((qTail + 1) % QDEPTH);

    // While idle, a frame must not drive the relays OR the LED. One can still
    // arrive if it was in flight when we stood down or reset, or if ESP-1 has
    // not heard our IDLE yet; the IDLE repeat stops it within 2 s. Discarding
    // it here is what keeps a fresh start / stand-down dark: no relay, no LED.
    if (!detectorRunning) continue;

    // ESP-1's frame IS its "I'm awake" signal: it only sends one after it has
    // really captured an image and run the model. The LED needs BOTH -- we asked
    // it to run (above) AND it proved it is running (this frame).
    lastGridAnyMs = millis();
    if (!esp1Awake) {
      esp1Awake = true;
      setStatusLed();
      Serial.println(F("\n>>> ESP-1 AWAKE: its frames are arriving -- status LED ON"));
    }

    if (!linkAlive) {
      Serial.println(F(">>> LINK UP: receiving grids from ESP-1"));
      linkAlive = true;
      lastSeq = 0;
    }
    lastRxMs = millis();
    totalRx++;
    if (lastSeq && m.seq > lastSeq + 1) totalDrops += (m.seq - lastSeq - 1);
    lastSeq = m.seq;

    // 1) fold this raw frame into the sliding window
    pushFrame(m.grid);

    // 2) apply the vote + hysteresis + dwell rule, zone by zone
    for (int z = 0; z < 9; z++) {
      bool want = verdict[z];                       // dead band -> hold
      if      (votes[z] >= ON_VOTES)  want = true;
      else if (votes[z] <= OFF_VOTES) want = false;

      if (want != verdict[z] && (millis() - lastChangeMs[z]) >= MIN_HOLD_MS) {
        verdict[z] = want;
        lastChangeMs[z] = millis();
        int a = ZONE_APPLIANCE[z];
        setAppliance(a, want);
        Serial.printf(">>> SWITCHED %s (zone %d) %s   [%u/%u frames occupied]\n",
                      appliances[a].label, z, want ? "ON" : "OFF",
                      votes[z], histFill);
      }
    }

    // 3) stand-down bookkeeping: count consecutive frames whose SMOOTHED
    //    verdict is fully empty. Any occupied zone resets the count. Counted
    //    before reporting so the printout shows the up-to-date streak.
    // Capped so it cannot wrap: while a motion hold blocks the stand-down the
    // count keeps climbing, and a uint16_t would roll over to 0 after ~13 h.
    if (roomLooksEmpty()) { if (emptyFrames < 60000) emptyFrames++; }
    else                  emptyFrames = 0;

    // 4) report: full block, or one compact line in quiet mode ('v')
    if (verbose) printFrame(m, src);
    else         printCompact(m);

    // 5) stand-down, unless recent PIR motion is holding it off
    if (emptyFrames >= EMPTY_FRAMES_TO_IDLE) {
      if (!motionHoldActive()) {
        stopDetector("room empty for 150 frames");
      } else if (emptyFrames == EMPTY_FRAMES_TO_IDLE) {
        // print once, when the count is first reached under a hold
        Serial.printf(">>> Room looks empty for %u frames, but PIR motion holds for %lu s more\n",
                      EMPTY_FRAMES_TO_IDLE,
                      (unsigned long)motionHoldLeftS());
      }
    }
  }

  // ---- keep ESP-1 in step: re-broadcast the current run/idle state ----
  if (millis() - lastCtrlSentMs >= CTRL_REPEAT_MS) sendControl();

  // ---- status LED: turn off once ESP-1 stops signalling that it is awake ----
  if (esp1Awake && (millis() - lastGridAnyMs) > ESP1_AWAKE_TIMEOUT_MS) {
    esp1Awake = false;
    setStatusLed();
    Serial.printf("\n>>> ESP-1 QUIET: no frame for %lu s -- status LED OFF (%s)\n",
                  (unsigned long)(ESP1_AWAKE_TIMEOUT_MS / 1000),
                  detectorRunning ? "UNEXPECTED: we asked it to run -- check ESP-1 power/link"
                                  : "expected: it was told to idle");
  }

  // ---- heartbeat while no grids are arriving, so silence is diagnosable ----
  // While idle, silence from ESP-1 is EXPECTED -- say so, rather than implying
  // the link is broken.
  static uint32_t lastBeat = 0;
  if (!linkAlive && (millis() - lastBeat) > 5000) {
    lastBeat = millis();
    if (detectorRunning) {
      uint8_t prim = 0; wifi_second_chan_t sec = WIFI_SECOND_CHAN_NONE;
      esp_wifi_get_channel(&prim, &sec);
      Serial.printf("[%lus] RUNNING, waiting for ESP-1 ... channel %u | raw callbacks: %lu | "
                    "rejected: %lu (last %d B) | accepted: %lu\n",
                    (unsigned long)(millis() / 1000), prim,
                    (unsigned long)rawCallbacks, (unsigned long)badLenCount,
                    lastBadLen, (unsigned long)totalRx);
    } else {
      Serial.printf("\n[%lus] IDLE -- ESP-1 stood down, all appliances off, waiting for PIR motion\n",
                    (unsigned long)(millis() / 1000));
    }
    // No frame blocks are printing now, so this heartbeat is the only regular
    // output -- carry the PIR and LED state in it too.
    printPirLine();
    printLedLine();
  }

  // ---- fail safe: ESP-1 went quiet WHILE IT SHOULD BE RUNNING ----
  // Gated on detectorRunning: when we have idled ESP-1 it stops sending on
  // purpose, and that must never be mistaken for a dead link.
  if (detectorRunning && linkAlive && (millis() - lastRxMs) > LINK_TIMEOUT_MS) {
    linkAlive = false;
    Serial.printf("\n>>> LINK LOST: no grid from ESP-1 for %lu ms while running -- all appliances OFF\n",
                  (unsigned long)LINK_TIMEOUT_MS);
    allOff();
    clearVerdicts();
    emptyFrames = 0;
  }

  delay(5);
}
