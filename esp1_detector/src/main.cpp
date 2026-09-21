// ESP-1 -- classroom occupancy detector.
//
// An ESP32-S3 with an OV3660 camera looks down at the room and reports which of
// nine floor zones are occupied. Each frame goes through:
//
//   capture 800x600 RGB565  ->  perspective warp to a canonical 96x96 top-down
//   view  ->  CLAHE contrast normalisation  ->  int8 MobileNetV2  ->  3x3 grid
//
// The result is broadcast over ESP-NOW to ESP-2, which smooths it and switches
// the fans and lights. This board makes no decisions about appliances; it only
// reports what it sees.
//
// The detector does not run continuously. ESP-2 owns a PIR motion sensor and
// tells this board when to work, so the camera and the model stay idle while
// the room is empty. See the control-message handling below.

#include <Arduino.h>
#include <math.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "esp_camera.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "occupancy_link.h"

// ---------------------------------------------------------------------------
// OV3660 wiring on this board
// ---------------------------------------------------------------------------
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y9_GPIO_NUM    16
#define Y8_GPIO_NUM    17
#define Y7_GPIO_NUM    18
#define Y6_GPIO_NUM    12
#define Y5_GPIO_NUM    10
#define Y4_GPIO_NUM     8
#define Y3_GPIO_NUM     9
#define Y2_GPIO_NUM    11
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

namespace {

// --- model / interpreter ---------------------------------------------------
const tflite::Model*      model       = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor*             input       = nullptr;
TfLiteTensor*             output      = nullptr;

// MobileNetV2 a0.35 at 96x96 in int8 needs about 320 KB of arena. 1.25 MiB is
// generous but PSRAM is plentiful, and the real figure is printed at boot.
constexpr int kTensorArenaSize = 1280 * 1024;
uint8_t* tensor_arena = nullptr;

// --- capture ---------------------------------------------------------------
// SVGA is the largest frame the OV3660 will hand over as raw RGB565; anything
// larger only comes out as JPEG, and decoding that on-chip costs seconds. Going
// straight to RGB565 removes the decode stage completely.
constexpr int   CAP_W = 800, CAP_H = 600;
// The homography below was measured against full-resolution 2048x1536 frames,
// so warp coordinates are scaled down into the SVGA frame.
constexpr float CAP_SCALE = (float)CAP_W / 2048.0f;
constexpr int   IN = 96;                  // model input is 96x96
// Pixels arriving straight from the sensor are byte-swapped relative to the
// little-endian order the decoder library produces.
constexpr bool  SWAP565 = true;
// The camera is mounted upside down relative to the images the model was
// trained on, so the sensor is told to flip vertically. Correcting it here
// rather than in software keeps the warp source in the expected orientation.
constexpr bool  CAM_VFLIP   = true;
constexpr bool  CAM_HMIRROR = false;

uint8_t* frame96 = nullptr;               // preprocessed image handed to the model

// Inverse homography mapping the canonical 96x96 view back into camera pixels,
// derived from the four measured floor corners of the monitored area.
const float HINV[9] = {
   12.60703244f, -6.22506448f, 431.11617264f,
    0.60589227f, -1.03000021f, 451.61136222f,
    0.0008839248f, -0.0067755600f, 1.00000000f
};

// CLAHE settings, matching cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
// used when preparing the training data.
constexpr int TILES = 8;
constexpr int TW = IN / TILES, TH = IN / TILES;

// --- link state ------------------------------------------------------------
const uint8_t BCAST[6] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
bool     linkReady = false;
uint32_t frameSeq  = 0;

// Set from the ESP-NOW receive callback, so it must be volatile.
volatile bool     runDetector   = true;   // assume "run" until told otherwise
volatile uint32_t lastCtrlMs    = 0;

// If the controller goes quiet for this long, fall back to running. Staying
// asleep because the other board died would silently disable the whole system,
// whereas running unnecessarily only wastes power.
constexpr uint32_t CTRL_TIMEOUT_MS = 30000;

// Idle polling interval. Nothing happens in this state except waiting for a
// control message, so there is no reason to spin quickly.
constexpr uint32_t IDLE_POLL_MS = 200;

// Gap between frames while running. A frame costs roughly 520 ms, so this gives
// about 0.72 s per result. ESP-2's smoothing windows are sized around that.
constexpr uint32_t LOOP_DELAY_MS = 200;

} // namespace

// ---------------------------------------------------------------------------
// Colour conversion, matching OpenCV's sRGB <-> CIE-Lab so that on-device
// preprocessing reproduces what the training pipeline did.
// ---------------------------------------------------------------------------
static inline float srgb2lin(float c) { return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); }
static inline float lin2srgb(float c) { return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f; }
static inline float labf(float t)     { return t > 0.008856f ? cbrtf(t) : 7.787f * t + 0.13793103f; }
static inline float labfi(float t)    { float t3 = t * t * t; return t3 > 0.008856f ? t3 : (t - 0.13793103f) / 7.787f; }

static const float Xn = 0.950456f, Zn = 1.088754f;

static inline uint8_t clamp8(float v) { return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)lroundf(v)); }

static void rgb2lab(uint8_t R, uint8_t G, uint8_t B, float& L, float& a, float& bb) {
  float r = srgb2lin(R / 255.f), g = srgb2lin(G / 255.f), b = srgb2lin(B / 255.f);
  float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / Xn;
  float Y =  0.212671f * r + 0.715160f * g + 0.072169f * b;
  float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / Zn;
  float fx = labf(X), fy = labf(Y), fz = labf(Z);
  L = 116.f * fy - 16.f; a = 500.f * (fx - fy); bb = 200.f * (fy - fz);
}

static void lab2rgb(float L, float a, float bb, uint8_t& R, uint8_t& G, uint8_t& B) {
  float fy = (L + 16.f) / 116.f, fx = fy + a / 500.f, fz = fy - bb / 200.f;
  float X = Xn * labfi(fx), Y = labfi(fy), Z = Zn * labfi(fz);
  float r =  3.240479f * X - 1.537150f * Y - 0.498535f * Z;
  float g = -0.969256f * X + 1.875992f * Y + 0.041556f * Z;
  float b =  0.055648f * X - 0.204043f * Y + 1.057311f * Z;
  R = clamp8(lin2srgb(r) * 255.f); G = clamp8(lin2srgb(g) * 255.f); B = clamp8(lin2srgb(b) * 255.f);
}

// ---------------------------------------------------------------------------
// CLAHE over the lightness channel, in place on a 96x96 RGB image.
//
// The classroom has windows down one side, so brightness across the floor
// varies a lot with the time of day. Equalising locally rather than globally
// keeps a sunlit zone and a shaded zone looking comparable to the model.
// ---------------------------------------------------------------------------
static void applyCLAHE(uint8_t* rgb) {
  static uint8_t Lc[IN * IN];
  static float   av[IN * IN], bv[IN * IN];

  for (int i = 0; i < IN * IN; i++) {
    float L, a, b;
    rgb2lab(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], L, a, b);
    Lc[i] = clamp8(L * 2.55f);
    av[i] = a;
    bv[i] = b;
  }

  // One histogram-equalisation lookup table per tile.
  static uint8_t lut[TILES][TILES][256];
  int clipLimit = (int)(2.0f * TW * TH / 256.0f);
  if (clipLimit < 1) clipLimit = 1;
  const float scale = 255.0f / (TW * TH);

  for (int ty = 0; ty < TILES; ty++) {
    for (int tx = 0; tx < TILES; tx++) {
      int hist[256] = { 0 };
      for (int y = 0; y < TH; y++)
        for (int x = 0; x < TW; x++)
          hist[Lc[(ty * TH + y) * IN + (tx * TW + x)]]++;

      // Clip tall bins and spread what was removed back across the histogram,
      // which is what stops CLAHE amplifying noise in flat areas.
      int excess = 0;
      for (int i = 0; i < 256; i++)
        if (hist[i] > clipLimit) { excess += hist[i] - clipLimit; hist[i] = clipLimit; }

      int inc = excess / 256, rem = excess - inc * 256;
      for (int i = 0; i < 256; i++) hist[i] += inc;
      if (rem > 0) {
        int step = 256 / rem > 0 ? 256 / rem : 1;
        for (int i = 0; i < 256 && rem > 0; i += step) { hist[i]++; rem--; }
      }

      int cdf = 0;
      for (int i = 0; i < 256; i++) {
        cdf += hist[i];
        float v = cdf * scale;
        lut[ty][tx][i] = v > 255 ? 255 : (uint8_t)lroundf(v);
      }
    }
  }

  // Bilinear blend between the four surrounding tile tables, so tile edges do
  // not show up as visible seams.
  for (int y = 0; y < IN; y++) {
    float gy = (float)y / TH - 0.5f;
    int ty0 = (int)floorf(gy);
    float wy = gy - ty0;
    if (ty0 < 0)          { ty0 = 0;         wy = 0; }
    if (ty0 > TILES - 1)  { ty0 = TILES - 1; wy = 0; }
    int ty1 = ty0 + 1 < TILES ? ty0 + 1 : ty0;

    for (int x = 0; x < IN; x++) {
      float gx = (float)x / TW - 0.5f;
      int tx0 = (int)floorf(gx);
      float wx = gx - tx0;
      if (tx0 < 0)         { tx0 = 0;         wx = 0; }
      if (tx0 > TILES - 1) { tx0 = TILES - 1; wx = 0; }
      int tx1 = tx0 + 1 < TILES ? tx0 + 1 : tx0;

      uint8_t v = Lc[y * IN + x];
      float top = lut[ty0][tx0][v] * (1 - wx) + lut[ty0][tx1][v] * wx;
      float bot = lut[ty1][tx0][v] * (1 - wx) + lut[ty1][tx1][v] * wx;
      float Lnew = (top * (1 - wy) + bot * wy) / 2.55f;

      int i = y * IN + x;
      lab2rgb(Lnew, av[i], bv[i], rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    }
  }
}

// Unpack one RGB565 pixel into 8-bit channels, replicating the high bits into
// the low ones so that full white maps to 255 rather than 248.
static inline void rd565(uint16_t p, int& r, int& g, int& b) {
  if (SWAP565) p = (uint16_t)((p >> 8) | (p << 8));
  r = ((p >> 11) & 0x1F) << 3; r |= r >> 5;
  g = ((p >>  5) & 0x3F) << 2; g |= g >> 6;
  b = ( p        & 0x1F) << 3; b |= b >> 5;
}

// ---------------------------------------------------------------------------
// Warp the camera frame into the canonical top-down 96x96 view.
// ---------------------------------------------------------------------------
static void warpFrame(const uint16_t* src16) {
  for (int oy = 0; oy < IN; oy++) {
    for (int ox = 0; ox < IN; ox++) {
      float w  = HINV[6] * ox + HINV[7] * oy + HINV[8];
      float fx = ((HINV[0] * ox + HINV[1] * oy + HINV[2]) / w) * CAP_SCALE;
      float fy = ((HINV[3] * ox + HINV[4] * oy + HINV[5]) / w) * CAP_SCALE;

      int o = (oy * IN + ox) * 3;
      if (fx < 0 || fx >= CAP_W - 1 || fy < 0 || fy >= CAP_H - 1) {
        frame96[o] = frame96[o + 1] = frame96[o + 2] = 0;   // outside the frame
        continue;
      }

      // Bilinear sample, matching cv2.warpPerspective with INTER_LINEAR.
      int x0 = (int)fx, y0 = (int)fy;
      float dx = fx - x0, dy = fy - y0;
      const uint16_t* row0 = &src16[(size_t)y0 * CAP_W + x0];
      const uint16_t* row1 = row0 + CAP_W;

      int r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
      rd565(row0[0], r00, g00, b00); rd565(row0[1], r10, g10, b10);
      rd565(row1[0], r01, g01, b01); rd565(row1[1], r11, g11, b11);

      float wr = (r00 * (1 - dx) + r10 * dx) * (1 - dy) + (r01 * (1 - dx) + r11 * dx) * dy;
      float wg = (g00 * (1 - dx) + g10 * dx) * (1 - dy) + (g01 * (1 - dx) + g11 * dx) * dy;
      float wb = (b00 * (1 - dx) + b10 * dx) * (1 - dy) + (b01 * (1 - dx) + b11 * dx) * dy;

      frame96[o]     = clamp8(wr);
      frame96[o + 1] = clamp8(wg);
      frame96[o + 2] = clamp8(wb);
    }
  }
}

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------
static bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_pwdn  = PWDN_GPIO_NUM;  c.pin_reset = RESET_GPIO_NUM; c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d0 = Y2_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.frame_size   = FRAMESIZE_SVGA;
  c.pixel_format = PIXFORMAT_RGB565;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  // Two buffers let the sensor fill one while the model works on the other, so
  // capture costs almost nothing once the pipeline is running.
  c.fb_count   = 2;
  c.grab_mode  = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&c) != ESP_OK) return false;

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, CAM_VFLIP ? 1 : 0);
    s->set_hmirror(s, CAM_HMIRROR ? 1 : 0);
  }
  return true;
}

// Handle a control message from ESP-2. Runs on the WiFi task, so it does as
// little as possible: validate, store, return.
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
static void onCtrlRecv(const esp_now_recv_info_t*, const uint8_t* data, int len) {
#else
static void onCtrlRecv(const uint8_t*, const uint8_t* data, int len) {
#endif
  if (len != (int)sizeof(ctrl_msg_t) || data[0] != MSG_CTRL) return;
  ctrl_msg_t m;
  memcpy(&m, data, sizeof(m));
  runDetector = (m.run != 0);
  lastCtrlMs  = millis();
}

static bool initEspNow() {
  WiFi.mode(WIFI_STA);          // needed for ESP-NOW; it never joins a network
  WiFi.disconnect();

  // The Arduino core brings the radio up in the background, and esp_wifi_*
  // calls made before it is ready fail without reporting anything useful. Wait
  // until the interface can produce its own MAC address before configuring it.
  uint8_t mac[6] = { 0 };
  for (int i = 0; i < 150; i++) {
    if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK &&
        (mac[0] | mac[1] | mac[2] | mac[3] | mac[4] | mac[5]) != 0) break;
    delay(20);
  }

  esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);

  // Read the channel back rather than trusting the call: a silent mismatch here
  // is indistinguishable from a dead link once the system is running.
  uint8_t chan = 0;
  wifi_second_chan_t sec = WIFI_SECOND_CHAN_NONE;
  esp_wifi_get_channel(&chan, &sec);
  if (chan != ESPNOW_CHANNEL) {
    Serial.printf("WARNING: radio is on channel %u, expected %u\n", chan, ESPNOW_CHANNEL);
  }

  if (esp_now_init() != ESP_OK) return false;
  esp_now_register_recv_cb(onCtrlRecv);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BCAST, 6);
  peer.channel = ESPNOW_CHANNEL;
  peer.encrypt = false;
  peer.ifidx   = WIFI_IF_STA;
  return esp_now_add_peer(&peer) == ESP_OK;
}

static void halt(const char* msg) {
  Serial.println(msg);
  while (true) delay(1000);
}

static size_t psramFree() { return heap_caps_get_free_size(MALLOC_CAP_SPIRAM); }

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  // Wait briefly for a serial host, but never indefinitely. On this board
  // Serial is the native USB port, and it only reports "ready" once a computer
  // opens it -- so waiting forever would stop the board ever starting when it
  // runs from a plain power supply.
  for (unsigned long t0 = millis(); !Serial && millis() - t0 < 1500; ) delay(10);
  delay(300);

  Serial.println("\n=== Classroom occupancy detector (ESP-1) ===");
  Serial.printf("PSRAM free: %u bytes\n", (unsigned)psramFree());

  if (!initCamera()) halt("FATAL: camera init failed");
  Serial.println("Camera ready: OV3660, 800x600 RGB565");

  // Over-allocate by 16 bytes so the arena can be aligned by hand; TFLite Micro
  // expects an aligned arena and heap_caps_malloc does not promise one.
  uint8_t* raw = (uint8_t*)heap_caps_malloc(kTensorArenaSize + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!raw) halt("FATAL: could not allocate tensor arena");
  tensor_arena = (uint8_t*)(((uintptr_t)raw + 15) & ~15);

  frame96 = (uint8_t*)heap_caps_malloc(IN * IN * 3, MALLOC_CAP_SPIRAM);
  if (!frame96) halt("FATAL: could not allocate frame buffer");

  // See model.cc: the SIMD kernels need this buffer 16-byte aligned, and a
  // misalignment produces wrong results rather than an error.
  if ((uintptr_t)model_tflite % 16 != 0)
    halt("FATAL: model buffer is not 16-byte aligned");

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt("FATAL: model schema mismatch");

  // Only the operators this network actually uses, which keeps the binary small.
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddLogistic();
  resolver.AddReshape();

  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &staticInterpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk)
    halt("FATAL: AllocateTensors failed -- increase kTensorArenaSize");

  input  = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("Model ready, arena in use: %u bytes\n", (unsigned)interpreter->arena_used_bytes());

  linkReady = initEspNow();
  Serial.print("MAC: ");
  Serial.println(WiFi.macAddress());
  if (linkReady) Serial.printf("ESP-NOW ready on channel %u\n", ESPNOW_CHANNEL);
  else           Serial.println("WARNING: ESP-NOW unavailable, results will not reach ESP-2");

  lastCtrlMs = millis();
}

void loop() {
  // --- idle state ----------------------------------------------------------
  // ESP-2 decides when this board should work. While idle the camera stays
  // initialised, which makes waking up immediate, but no frame is captured and
  // the model never runs.
  if (!runDetector) {
    // Do not stay asleep if the controller has stopped talking to us: a dead or
    // rebooting ESP-2 must not leave the room permanently unmonitored.
    if (millis() - lastCtrlMs > CTRL_TIMEOUT_MS) {
      Serial.println("No control messages received -- resuming detection");
      runDetector = true;
      lastCtrlMs  = millis();
    }
    delay(IDLE_POLL_MS);
    return;
  }

  // --- capture -------------------------------------------------------------
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed");
    delay(500);
    return;
  }

  // The warp reads the camera's buffer directly, so there is no intermediate copy.
  warpFrame((const uint16_t*)fb->buf);
  esp_camera_fb_return(fb);

  applyCLAHE(frame96);

  // --- quantise into the input tensor --------------------------------------
  // Pixels are scaled to [-1, 1] to match training, then mapped onto the
  // model's int8 quantisation range.
  const float iscale = input->params.scale;
  const int   izp    = input->params.zero_point;
  for (int i = 0; i < IN * IN * 3; i++) {
    int q = (int)lroundf((frame96[i] / 127.5f - 1.0f) / iscale) + izp;
    input->data.int8[i] = (int8_t)constrain(q, -128, 127);
  }

  unsigned long t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    delay(500);
    return;
  }
  unsigned long inferMs = millis() - t0;

  // --- read the 3x3 grid ---------------------------------------------------
  const float osc = output->params.scale;
  const int   ozp = output->params.zero_point;

  uint8_t grid[9];
  int occupied = 0;
  for (int i = 0; i < 9; i++) {
    float p = (output->data.int8[i] - ozp) * osc;   // sigmoid probability
    grid[i] = (p > 0.5f) ? 1 : 0;
    occupied += grid[i];
  }

  Serial.printf("grid %d%d%d %d%d%d %d%d%d  occupied %d/9  infer %lu ms  seq %lu\n",
                grid[0], grid[1], grid[2], grid[3], grid[4], grid[5],
                grid[6], grid[7], grid[8], occupied, inferMs,
                (unsigned long)(frameSeq + 1));

  // --- publish -------------------------------------------------------------
  if (linkReady) {
    grid_msg_t msg;
    msg.tag      = MSG_GRID;
    memcpy(msg.grid, grid, sizeof(grid));
    msg.occupied = (uint8_t)occupied;
    msg.infer_ms = (uint16_t)inferMs;
    msg.seq      = ++frameSeq;
    esp_now_send(BCAST, (const uint8_t*)&msg, sizeof(msg));
  }

  delay(LOOP_DELAY_MS);
}
