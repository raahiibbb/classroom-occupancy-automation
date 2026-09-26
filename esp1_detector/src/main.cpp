// Standalone classroom occupancy detector — SVGA direct-RGB565 edition.
// Pipeline: OV3660 SVGA 800x600 DIRECT RGB565 capture (no JPEG, no decode) -> perspective WARP
// (BILINEAR, matching cv2.warpPerspective INTER_LINEAR) to canonical 3x3 view -> CLAHE
// (L channel) -> normalize [-1,1] -> int8 MobileNetV2 (ESP-NN) -> 3x3 occupancy grid.
//
// This captures at the SENSOR'S MAX resolution — the SAME 2048x1536 the training/day-3
// images were shot at — so the 96x96 fed to the model matches the PC eval distribution
// (the previous build captured only QVGA 320x240 -> ~41x fewer pixels -> train/serve skew).
// The warp source is now the full 2048x1536 frame, so HINV uses the raw grid_config corners
// (no QVGA scaling). Memory: the 6.0 MiB RGB565 frame + ~1.25 MiB arena live in 8 MB PSRAM.
//   'D' over serial -> dumps the final 96x96 fed to the model (post warp+CLAHE)
#include <Arduino.h>
#include <math.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
// --- ESP-NOW link to the relay/actuator board (ESP-2) ---
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ---------------------------------------------------------------------------
// ESP-NOW broadcast of the 3x3 grid to ESP-2 (classic ESP32 relay board).
// Broadcast (FF:..:FF) needs no MAC pairing: ESP-2 just listens on the same
// channel. This struct MUST match the receiver sketch byte for byte.
// ---------------------------------------------------------------------------
// Wire format (grid_msg_t / ctrl_msg_t / MSG_GRID / MSG_CTRL / ESPNOW_CHANNEL)
// is shared with ESP-2 via this header. The two copies -- include/ here and the
// one beside the Arduino sketch -- MUST stay byte-for-byte identical, or the
// tag and length checks on each side silently drop every packet.
#include "occupancy_link.h"

static const uint8_t BCAST[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
static bool     g_espnow_ok = false;
static uint32_t g_seq = 0;

// --- run / idle, commanded by ESP-2 -----------------------------------------
// ESP-2 owns this decision: it has the PIR and the smoothed view of the room.
//
// FRESH START: every power-up or reset begins IDLE -- room assumed empty, no
// capture, no inference, no grids sent. Only a RUN from ESP-2 (i.e. PIR motion
// there) wakes us.
static volatile bool     runDetector = false;
static volatile uint32_t lastCtrlMs  = 0;

// Boot sync. Until ESP-2 has acknowledged this boot by sending an IDLE, every
// RUN is ignored: a RUN arriving now could be a stale one ESP-2 was repeating
// from BEFORE we rebooted, and obeying it would skip the fresh start. We send
// HELLO (see occupancy_link.h) every HELLO_REPEAT_MS until that IDLE arrives;
// ESP-2 responds by resetting the whole system to fresh idle.
static volatile bool     bootSynced   = false;
static uint32_t          helloSeq     = 0;
static uint32_t          lastHelloMs  = 0;
static const uint32_t    HELLO_REPEAT_MS = 1000;

// If ESP-2 goes quiet this long WHILE WE ARE RUNNING, go idle. The relays live
// on ESP-2, so running without it achieves nothing -- and a silent ESP-2 has
// most likely been powered off, which should leave the system idle, not busy.
// (This replaces the old rule of RESUMING after 30 s of silence, which
// contradicted the fresh-start behaviour.) ESP-2 repeats its state every 2 s,
// so 10 s is five lost packets in a row before this can trigger.
static const uint32_t CTRL_LOST_IDLE_MS = 10000;
// How often to re-check the control state while idle (camera stays initialised
// so waking is immediate, but no capture and no inference happen).
static const uint32_t IDLE_POLL_MS = 100;
// Loop pacing. A frame costs ~500 ms (SVGA warp+CLAHE+infer), so 200 ms idle
// gives ~0.7 s per grid; ESP-2 smooths 20 of them => a ~14 s decision window.
static const uint32_t LOOP_DELAY_MS = 200;

// --- OV3660 pin map (unchanged) ---
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  // MobileNetV2 a0.35 @96x96 int8 peaks well under 0.5 MiB; 1.25 MiB leaves the rest of
  // PSRAM for the 6.0 MiB full-res frame. arena_used_bytes() is printed at boot to confirm.
  constexpr int kTensorArenaSize = 1280 * 1024;
  uint8_t* tensor_arena = nullptr;
  // DIRECT RGB565 capture at SVGA 800x600 — NO JPEG, NO on-chip decode. The old QXGA JPEG
  // path's ~2.2 s jpg2rgb565 decode (TJpgDec in ROM: sequential Huffman, so JPG_SCALE_* can't
  // cut it — measured ~0 win) was the ENTIRE end-to-end bottleneck. SVGA is the largest frame
  // the OV3660 delivers as direct RGB565 (SXGA+ fail to grab); this removes decode outright.
  // Held-out ("day-3") re-evaluation on PC at 800x600 = 87.66% per-cell (-0.96 vs
  // full-res 88.62), AUC 92.80. fb_count=2 lets capture overlap compute -> ~0 ms/frame.
  constexpr int CAP_W = 800, CAP_H = 600;                 // SVGA — sensor delivers RGB565 here
  constexpr float CAP_SCALE = (float)CAP_W / 2048.0f;     // full-res HINV coord -> SVGA (4:3, uniform)
  constexpr int IN = 96;
  constexpr bool SWAP565 = true;             // DIRECT-sensor RGB565 IS byte-swapped vs LE (unlike jpg2rgb565)
  // Sensor orientation — must match the day-3 dataset HINV was calibrated on. The OV3660
  // live feed comes out vertically flipped vs day-3, so flip it back at the sensor (register
  // level), which keeps the raw warp source correct. Toggle these if live capture shows the
  // view is still upside-down (CAM_VFLIP) or left/right swapped (CAM_HMIRROR).
  constexpr bool CAM_VFLIP   = true;         // 1 = top/bottom flip (matches day-3)
  constexpr bool CAM_HMIRROR = false;        // 1 = left/right mirror (none observed on day-3)
  uint8_t* frame96 = nullptr;                // final preprocessed RGB fed to the model (for 'D' dump)

  // canonical(96x96) -> camera(2048x1536) inverse homography, from day-3 grid_config corners
  // [[547,465],[1418,479],[2044,889],[3,923]] (FULL-RES, unscaled). Recompute these nine
  // values if the camera is moved -- see "Recalibrating after moving the camera" in README.md.
  const float HINV[9] = {
     12.60703244f, -6.22506448f, 431.11617264f,
     0.60589227f, -1.03000021f, 451.61136222f,
     0.0008839248f, -0.0067755600f, 1.00000000f
  };

  // ---- CLAHE params (must match cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))) ----
  constexpr int TILES = 8;
  constexpr int TW = IN / TILES, TH = IN / TILES;
}

// ---------- sRGB <-> CIE-LAB (D65), matching OpenCV coefficients ----------
static inline float srgb2lin(float c) { return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); }
static inline float lin2srgb(float c) { return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f; }
static inline float labf(float t)    { return t > 0.008856f ? cbrtf(t) : 7.787f * t + 0.13793103f; }
static inline float labfi(float t)   { float t3 = t * t * t; return t3 > 0.008856f ? t3 : (t - 0.13793103f) / 7.787f; }
static const float Xn = 0.950456f, Zn = 1.088754f;

static void rgb2lab(uint8_t R, uint8_t G, uint8_t B, float& L, float& a, float& bb) {
  float r = srgb2lin(R / 255.f), g = srgb2lin(G / 255.f), b = srgb2lin(B / 255.f);
  float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / Xn;
  float Y =  0.212671f * r + 0.715160f * g + 0.072169f * b;
  float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / Zn;
  float fx = labf(X), fy = labf(Y), fz = labf(Z);
  L = 116.f * fy - 16.f; a = 500.f * (fx - fy); bb = 200.f * (fy - fz);
}
static inline uint8_t clamp8(float v) { return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)lroundf(v)); }
static void lab2rgb(float L, float a, float bb, uint8_t& R, uint8_t& G, uint8_t& B) {
  float fy = (L + 16.f) / 116.f, fx = fy + a / 500.f, fz = fy - bb / 200.f;
  float X = Xn * labfi(fx), Y = labfi(fy), Z = Zn * labfi(fz);
  float r =  3.240479f * X - 1.537150f * Y - 0.498535f * Z;
  float g = -0.969256f * X + 1.875992f * Y + 0.041556f * Z;
  float b =  0.055648f * X - 0.204043f * Y + 1.057311f * Z;
  R = clamp8(lin2srgb(r) * 255.f); G = clamp8(lin2srgb(g) * 255.f); B = clamp8(lin2srgb(b) * 255.f);
}

// ---------- CLAHE on the L channel of a 96x96 RGB image (in place) ----------
static void applyCLAHE(uint8_t* rgb) {
  static uint8_t Lc[IN * IN];
  static float av[IN * IN], bv[IN * IN];
  for (int i = 0; i < IN * IN; i++) {
    float L, a, b;
    rgb2lab(rgb[i*3], rgb[i*3+1], rgb[i*3+2], L, a, b);
    Lc[i] = clamp8(L * 2.55f); av[i] = a; bv[i] = b;
  }
  static uint8_t lut[TILES][TILES][256];
  int clipLimit = (int)(2.0f * TW * TH / 256.0f); if (clipLimit < 1) clipLimit = 1;
  const float scale = 255.0f / (TW * TH);
  for (int ty = 0; ty < TILES; ty++) for (int tx = 0; tx < TILES; tx++) {
    int hist[256] = {0};
    for (int y = 0; y < TH; y++) for (int x = 0; x < TW; x++)
      hist[Lc[(ty*TH + y) * IN + (tx*TW + x)]]++;
    int excess = 0;
    for (int i = 0; i < 256; i++) if (hist[i] > clipLimit) { excess += hist[i] - clipLimit; hist[i] = clipLimit; }
    int inc = excess / 256, rem = excess - inc * 256;
    for (int i = 0; i < 256; i++) hist[i] += inc;
    if (rem > 0) { int step = 256 / rem > 0 ? 256 / rem : 1; for (int i = 0; i < 256 && rem > 0; i += step) { hist[i]++; rem--; } }
    int cdf = 0;
    for (int i = 0; i < 256; i++) { cdf += hist[i]; float v = cdf * scale; lut[ty][tx][i] = v > 255 ? 255 : (uint8_t)lroundf(v); }
  }
  for (int y = 0; y < IN; y++) {
    float gy = (float)y / TH - 0.5f; int ty0 = (int)floorf(gy); float wy = gy - ty0;
    if (ty0 < 0) { ty0 = 0; wy = 0; } if (ty0 > TILES - 1) { ty0 = TILES - 1; wy = 0; }
    int ty1 = ty0 + 1 < TILES ? ty0 + 1 : ty0;
    for (int x = 0; x < IN; x++) {
      float gx = (float)x / TW - 0.5f; int tx0 = (int)floorf(gx); float wx = gx - tx0;
      if (tx0 < 0) { tx0 = 0; wx = 0; } if (tx0 > TILES - 1) { tx0 = TILES - 1; wx = 0; }
      int tx1 = tx0 + 1 < TILES ? tx0 + 1 : tx0;
      uint8_t v = Lc[y * IN + x];
      float top = lut[ty0][tx0][v] * (1 - wx) + lut[ty0][tx1][v] * wx;
      float bot = lut[ty1][tx0][v] * (1 - wx) + lut[ty1][tx1][v] * wx;
      float Lnew = (top * (1 - wy) + bot * wy) / 2.55f;
      int i = y * IN + x;
      lab2rgb(Lnew, av[i], bv[i], rgb[i*3], rgb[i*3+1], rgb[i*3+2]);
    }
  }
}

// ---------- decode one RGB565 pixel (LCD byte order) -> 8-bit R,G,B ----------
static inline void rd565(uint16_t p, int& r, int& g, int& b) {
  if (SWAP565) p = (uint16_t)((p >> 8) | (p << 8));
  r = ((p >> 11) & 0x1F) << 3; r |= r >> 5;
  g = ((p >> 5)  & 0x3F) << 2; g |= g >> 6;
  b = ( p        & 0x1F) << 3; b |= b >> 5;
}

static bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0; c.ledc_timer = LEDC_TIMER_0;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM; c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d0 = Y2_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.frame_size = FRAMESIZE_SVGA;             // 800x600 — max res the OV3660 delivers as direct RGB565
  c.pixel_format = PIXFORMAT_RGB565;         // DIRECT sensor RGB565 — no JPEG, no on-chip decode
  c.fb_location = CAMERA_FB_IN_PSRAM; c.fb_count = 2; c.grab_mode = CAMERA_GRAB_LATEST;  // fb=2 overlaps capture w/ compute
  if (esp_camera_init(&c) != ESP_OK) return false;
  // Match day-3 orientation at the sensor so the raw QXGA frame HINV warps is correct.
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, CAM_VFLIP ? 1 : 0);
    s->set_hmirror(s, CAM_HMIRROR ? 1 : 0);
  }
  return true;
}
// Control message from ESP-2. Runs on the WiFi task, so it validates and
// stores only. Tag AND length are both checked so a stray ESP-NOW packet from
// some other device on this channel cannot steer us.
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
static void onCtrlRecv(const esp_now_recv_info_t*, const uint8_t* data, int len) {
#else
static void onCtrlRecv(const uint8_t*, const uint8_t* data, int len) {
#endif
  if (len != (int)sizeof(ctrl_msg_t) || data[0] != MSG_CTRL) return;
  ctrl_msg_t m;
  memcpy(&m, data, sizeof(m));
  lastCtrlMs = millis();

  if (!bootSynced) {
    // Fresh start: the first IDLE proves ESP-2 has seen our HELLO and reset
    // itself. Until then a RUN may be stale (from before our reboot) -- ignore it.
    if (m.run == 0) { bootSynced = true; runDetector = false; }
    return;
  }
  runDetector = (m.run != 0);
}

// Tell ESP-2 we have just (re)started, so it resets the whole system to idle.
static void sendHello() {
  hello_msg_t h;
  h.tag = MSG_HELLO;
  h.seq = ++helloSeq;
  esp_now_send(BCAST, (const uint8_t*)&h, sizeof(h));
  lastHelloMs = millis();
}

// Bring WiFi up in STA mode (it never associates to an AP) and register the
// broadcast peer. Failure is non-fatal: the detector still runs and prints to
// serial, it just cannot feed ESP-2.
static bool initEspNow() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // Arduino core 3.x starts the radio ASYNCHRONOUSLY; esp_wifi_* calls made
  // before it is up fail silently (and the channel never gets set). Wait for a
  // real MAC first, then configure, then VERIFY by reading the channel back --
  // printing the ESPNOW_CHANNEL constant proves nothing.
  uint8_t mac[6] = {0};
  for (int i = 0; i < 150; i++) {
    if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK &&
        (mac[0] | mac[1] | mac[2] | mac[3] | mac[4] | mac[5]) != 0) break;
    delay(20);
  }
  esp_err_t ce = esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  uint8_t prim = 0; wifi_second_chan_t sec = WIFI_SECOND_CHAN_NONE;
  esp_wifi_get_channel(&prim, &sec);
  Serial.printf("set_channel(%d) -> %s | radio is ACTUALLY on channel %u\n",
                ESPNOW_CHANNEL, esp_err_to_name(ce), prim);
  Serial.printf("sizeof(grid_msg_t) = %u, ctrl_msg_t = %u, hello_msg_t = %u bytes (expect 17/6/5)\n",
                (unsigned)sizeof(grid_msg_t), (unsigned)sizeof(ctrl_msg_t),
                (unsigned)sizeof(hello_msg_t));
  if (prim != ESPNOW_CHANNEL)
    Serial.println("*** WARNING: radio channel != ESPNOW_CHANNEL - ESP-2 will hear nothing ***");

  esp_err_t ne = esp_now_init();
  Serial.printf("esp_now_init() -> %s\n", esp_err_to_name(ne));
  if (ne != ESP_OK) return false;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BCAST, 6);
  peer.channel = ESPNOW_CHANNEL;
  peer.encrypt = false;
  peer.ifidx   = WIFI_IF_STA;
  esp_err_t pe = esp_now_add_peer(&peer);
  Serial.printf("esp_now_add_peer(broadcast) -> %s\n", esp_err_to_name(pe));

  // The link is now two-way: ESP-2 tells us when to run.
  esp_err_t re = esp_now_register_recv_cb(onCtrlRecv);
  Serial.printf("esp_now_register_recv_cb() -> %s\n", esp_err_to_name(re));

  return pe == ESP_OK;
}

static void halt(const char* m) { Serial.println(m); while (1) delay(1000); }
static size_t psram_free() { return heap_caps_get_free_size(MALLOC_CAP_SPIRAM); }

void setup() {
  Serial.begin(115200);
  // BOUNDED wait. With ARDUINO_USB_CDC_ON_BOOT=1, Serial is the native USB CDC and
  // `!Serial` stays true until a USB HOST opens the port. A charger or bench supply
  // never does, so an unbounded `while (!Serial)` hangs the board in setup() and it
  // never detects anything. Writes with no host attached are safely discarded by
  // HWCDC::write(), so running headless is fine -- we just must not wait forever.
  for (unsigned long t0 = millis(); !Serial && millis() - t0 < 1500; ) delay(10);
  delay(300);
  Serial.println("\n--- ESP32-S3 OCCUPANCY DETECTOR (SVGA direct-RGB565 warp+CLAHE) ---");
  Serial.printf("PSRAM free at boot: %u bytes\n", (unsigned)psram_free());
  if (!initCamera()) halt("FATAL: camera init failed");
  Serial.printf("Camera OK (OV3660, SVGA 800x600 direct RGB565, no decode). PSRAM free: %u\n", (unsigned)psram_free());

  uint8_t* raw = (uint8_t*)heap_caps_malloc(kTensorArenaSize + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!raw) halt("FATAL: arena alloc failed");
  tensor_arena = (uint8_t*)(((uintptr_t)raw + 15) & ~15);

  // No decode buffer needed — direct RGB565 frames come straight from the camera's own
  // PSRAM framebuffers (fb->buf); the warp samples them in place.
  frame96 = (uint8_t*)heap_caps_malloc(IN * IN * 3, MALLOC_CAP_SPIRAM);
  if (!frame96) halt("FATAL: frame96 alloc failed");
  Serial.printf("Buffers allocated. PSRAM free: %u\n", (unsigned)psram_free());

  // GUARD: TFLite Micro reads tensor data in place from this flatbuffer, and the
  // ESP-NN ESP32-S3 SIMD kernels require 16-byte alignment. model.cc declares the
  // array alignas(16); if that is ever lost, the SIMD kernels silently return
  // SATURATED garbage (every cell 0.996 -> 9/9 occupied) at normal speed, with a
  // perfectly good input image. Linking WiFi shifts .rodata and is what exposed it.
  Serial.printf("model_tflite @ %p  (16-byte aligned: %s)\n",
                model_tflite, ((uintptr_t)model_tflite % 16 == 0) ? "yes" : "NO");
  if ((uintptr_t)model_tflite % 16 != 0)
    Serial.println("*** WARNING: model buffer misaligned - ESP-NN output will be WRONG ***");

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt("FATAL: schema mismatch");
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddAdd(); resolver.AddLogistic(); resolver.AddReshape();
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) halt("FATAL: AllocateTensors failed — raise kTensorArenaSize");
  input = interpreter->input(0); output = interpreter->output(0);
  Serial.printf("Model ready (input %s). arena_used=%u bytes.\n",
                input->type == kTfLiteInt8 ? "int8" : "float32", (unsigned)interpreter->arena_used_bytes());

  g_espnow_ok = initEspNow();
  Serial.print("Sender MAC: "); Serial.println(WiFi.macAddress());
  lastCtrlMs = millis();
  if (g_espnow_ok) Serial.printf("ESP-NOW ready on channel %d\n", ESPNOW_CHANNEL);

  // Fresh start: begin IDLE, and announce the reboot so ESP-2 resets too.
  Serial.println("FRESH START: idle -- no capture, no grids -- until ESP-2 sends RUN (PIR motion).");
  if (g_espnow_ok) {
    Serial.println("Sending HELLO to ESP-2 every 1 s until it acknowledges with IDLE ...");
    sendHello();
  }
  else             Serial.println("WARNING: ESP-NOW init FAILED - serial output only");
}

void loop() {
  // --- boot sync with ESP-2 ------------------------------------------------
  // Keep announcing the reboot until ESP-2 acknowledges it with an IDLE.
  static bool announcedSync = false;
  if (!bootSynced) {
    if (g_espnow_ok && millis() - lastHelloMs >= HELLO_REPEAT_MS) sendHello();
  } else if (!announcedSync) {
    announcedSync = true;
    Serial.printf("\n>>> SYNCED with ESP-2 after %lu HELLO(s) -- system reset to fresh idle\n",
                  (unsigned long)helloSeq);
  }

  // --- ESP-2 went silent while we were running -> go idle -------------------
  if (runDetector && millis() - lastCtrlMs > CTRL_LOST_IDLE_MS) {
    Serial.printf("\n>>> No control message from ESP-2 for %lu s -- going IDLE\n",
                  (unsigned long)(CTRL_LOST_IDLE_MS / 1000));
    runDetector = false;
  }

  // --- run / idle ----------------------------------------------------------
  // ESP-2 owns this decision: it has the PIR and the smoothed view of the room.
  // While idle the camera and radio stay initialised, so waking is immediate,
  // but no frame is captured, the model never runs and no grid is sent.
  static bool wasRunning = false;          // fresh start = idle
  if (!runDetector) {
    if (wasRunning) {
      Serial.println("\n>>> IDLE: capture + inference stopped, no grids sent");
      wasRunning = false;
    }
    while (Serial.available()) Serial.read();   // do not let the buffer stagnate
    delay(IDLE_POLL_MS);
    return;
  }
  if (!wasRunning) {
    Serial.println("\n>>> RUNNING: ESP-2 woke the detector");
    wasRunning = true;
  }

  char cmd = 0;
  while (Serial.available()) { char ch = Serial.read(); if (ch == 'D' || ch == 'B') cmd = ch; }

  unsigned long tA = millis();
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { Serial.println("capture failed"); delay(500); return; }
  unsigned long tB = millis();
  unsigned long tC = tB;                       // no decode stage in the direct-RGB565 path

  // WARP source is the camera's RGB565 framebuffer itself (no decode/copy).
  const uint16_t* src16 = (const uint16_t*)fb->buf;

  // 2) WARP (bilinear): canonical 96x96 <- camera SVGA 800x600 via inverse homography
  for (int oy = 0; oy < IN; oy++) for (int ox = 0; ox < IN; ox++) {
    float w  = HINV[6]*ox + HINV[7]*oy + HINV[8];
    // HINV maps to FULL-RES (2048x1536) camera coords; rescale into the SVGA frame.
    float fx = ((HINV[0]*ox + HINV[1]*oy + HINV[2]) / w) * CAP_SCALE;
    float fy = ((HINV[3]*ox + HINV[4]*oy + HINV[5]) / w) * CAP_SCALE;
    int o = (oy*IN + ox) * 3;
    if (fx < 0 || fx >= CAP_W - 1 || fy < 0 || fy >= CAP_H - 1) { frame96[o]=frame96[o+1]=frame96[o+2]=0; continue; }
    int x0 = (int)fx, y0 = (int)fy; float dx = fx - x0, dy = fy - y0;
    const uint16_t* row0 = &src16[(size_t)y0 * CAP_W + x0];
    const uint16_t* row1 = row0 + CAP_W;
    int r00,g00,b00, r10,g10,b10, r01,g01,b01, r11,g11,b11;
    rd565(row0[0], r00,g00,b00); rd565(row0[1], r10,g10,b10);
    rd565(row1[0], r01,g01,b01); rd565(row1[1], r11,g11,b11);
    float wr = (r00*(1-dx)+r10*dx)*(1-dy) + (r01*(1-dx)+r11*dx)*dy;
    float wg = (g00*(1-dx)+g10*dx)*(1-dy) + (g01*(1-dx)+g11*dx)*dy;
    float wb = (b00*(1-dx)+b10*dx)*(1-dy) + (b01*(1-dx)+b11*dx)*dy;
    frame96[o]   = clamp8(wr);
    frame96[o+1] = clamp8(wg);
    frame96[o+2] = clamp8(wb);
  }
  esp_camera_fb_return(fb);                     // done sampling the framebuffer

  // 3) CLAHE lighting-normalization (in place on frame96)
  applyCLAHE(frame96);
  unsigned long tD = millis();

  // 4) normalize [-1,1] -> int8 into the input tensor
  const bool isInt8 = (input->type == kTfLiteInt8);
  const float iscale = input->params.scale; const int izp = input->params.zero_point;
  for (int i = 0; i < IN * IN * 3; i++) {
    if (isInt8) input->data.int8[i] = (int8_t)constrain((int)lroundf((frame96[i]/127.5f - 1.0f)/iscale) + izp, -128, 127);
    else input->data.f[i] = (float)frame96[i];
  }

  if (cmd == 'D') { Serial.println("FRAMESTART"); Serial.write(frame96, IN*IN*3); Serial.println(); Serial.println("FRAMEEND"); }
  if (cmd == 'B') {   // lossless HEX dump of frame96 (robust over flaky serial)
    static char hexbuf[IN*IN*3*2 + 1];
    static const char* HX = "0123456789abcdef";
    for (int i = 0; i < IN*IN*3; i++) { hexbuf[i*2] = HX[frame96[i] >> 4]; hexbuf[i*2+1] = HX[frame96[i] & 0xF]; }
    Serial.println("HEXSTART");
    Serial.write((const uint8_t*)hexbuf, IN*IN*3*2);
    Serial.println(); Serial.println("HEXEND");
  }

  unsigned long tE = millis();
  if (interpreter->Invoke() != kTfLiteOk) { Serial.println("invoke failed"); delay(500); return; }
  unsigned long tF = millis();

  const bool oInt8 = (output->type == kTfLiteInt8);
  const float osc = output->params.scale; const int ozp = output->params.zero_point;
  uint8_t grid[9];
  int occ = 0;
  Serial.printf("TIMING capture=%lu decode=%lu warp_clahe=%lu infer=%lu | TOTAL=%lu ms | 3x3 grid:\n",
                tB-tA, tC-tB, tD-tC, tF-tE, tF-tA);
  for (int i = 0; i < 9; i++) {
    float v = oInt8 ? (output->data.int8[i] - ozp) * osc : output->data.f[i];
    int pred = (v > 0.5f) ? 1 : 0; occ += pred;
    grid[i] = (uint8_t)pred;
    Serial.print(pred); Serial.print((i % 3 == 2) ? "\n" : " ");
  }
  Serial.print("SIGMOIDS:");
  for (int i = 0; i < 9; i++) {
    float v = (output->type == kTfLiteInt8)
            ? (output->data.int8[i] - output->params.zero_point) * output->params.scale
            : output->data.f[i];
    Serial.printf(" %.3f", v);
  }
  Serial.println();
  Serial.printf("occupied cells: %d/9\n", occ);
  // Broadcast this RAW (unsmoothed) frame to ESP-2, which does the temporal
  // thresholding and drives the relays.
  if (g_espnow_ok) {
    grid_msg_t msg;
    msg.tag = MSG_GRID;
    memcpy(msg.grid, grid, 9);
    msg.occupied = (uint8_t)occ;
    msg.infer_ms = (uint16_t)(tF - tE);
    msg.seq      = ++g_seq;
    esp_err_t r = esp_now_send(BCAST, (const uint8_t*)&msg, sizeof(msg));
    Serial.printf("ESP-NOW seq %lu %s\n", (unsigned long)msg.seq,
                  r == ESP_OK ? "sent" : "SEND FAILED");
  }
  Serial.println();

  delay(LOOP_DELAY_MS);
}
