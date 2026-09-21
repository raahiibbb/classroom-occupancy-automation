# Classroom Occupancy Automation

Two ESP32 boards that switch a classroom's fans and lights based on where people
actually are, rather than on a single switch for the whole room.

A ceiling-mounted camera board runs a small neural network that divides the floor
into a 3×3 grid and decides which zones are occupied. A second board receives
those readings over a wireless link, smooths them, and drives one relay per zone.
A PIR motion sensor lets the whole system stand down while the room is empty.

```
        ┌──────────────────────────┐                  ┌───────────────────────────┐
        │  ESP-1  ESP32-S3 + OV3660│   grid, ~1.4 Hz  │  ESP-2  ESP32 NodeMCU     │
        │                          │ ───────────────► │                           │
        │  capture → warp → CLAHE  │                  │  smoothing + thresholds   │
        │  → int8 CNN → 3×3 grid   │ ◄─────────────── │  relay control            │
        └──────────────────────────┘   run / idle     └───────────┬───────────────┘
                                                                  │
                                                    ┌─────────────┴─────────────┐
                                                    │                           │
                                              HC-SR501 PIR            4 fans + 5 bulbs
```

## Repository layout

| Path | Contents |
|------|----------|
| `esp1_detector/` | PlatformIO project for the camera board |
| `esp2_controller/` | Arduino sketch for the relay board |
| `model/` | The trained TensorFlow Lite model |
| `*/occupancy_link.h` | Message format shared by both boards |

## How it works

### Detection (ESP-1)

Each frame runs through four stages:

1. **Capture** — 800×600 RGB565 straight from the OV3660. This is the largest
   frame the sensor will provide as raw pixels; anything larger arrives as JPEG,
   and decoding that on the chip costs seconds per frame.
2. **Perspective warp** — the camera views the floor at an angle, so a
   homography maps the four measured floor corners onto a square, top-down
   96×96 image. The model then sees the same geometry regardless of where the
   camera is mounted.
3. **CLAHE** — contrast is equalised in small tiles rather than globally. The
   room has windows down one side, so a sunlit zone and a shaded zone would
   otherwise look very different to the network.
4. **Inference** — a quantised MobileNetV2 outputs nine probabilities, one per
   zone, thresholded at 0.5.

Inference takes about 200 ms using the ESP-NN optimised kernels, and a complete
cycle takes roughly 0.7 s.

### Switching (ESP-2)

The raw per-frame output is not stable enough to drive relays directly —
individual frames flip state fairly often even when nobody is moving. Each zone
therefore votes over a sliding window:

| Rule | Value | Meaning |
|------|-------|---------|
| Window | 20 frames | about 14 seconds of history |
| Switch on | ≥ 12 of 20 | zone occupied in 60% of recent frames |
| Switch off | ≤ 6 of 20 | zone occupied in 30% or fewer |
| Between the two | — | hold the current state |
| Minimum hold | 4 s | a zone cannot change state faster than this |

Two thresholds rather than one is the important part. With a single cut-off, a
zone sitting near the boundary would switch on and off repeatedly; the gap
between the thresholds removes that entirely.

### Standing down when the room is empty

The detector does not run continuously. ESP-2 decides when ESP-1 should work:

| Rule | Value | Meaning |
|------|-------|---------|
| Idle after | 150 consecutive empty frames | roughly 1.8 minutes of an empty room |
| Wake on | any PIR motion | after the sensor's warm-up period |
| Motion keeps it awake for | 60 s after any motion | see below |
| PIR warm-up | 60 s from power-up | the HC-SR501 reports spurious motion until it settles |

The empty-room count uses the **smoothed** verdict, not the raw frames, so a
single false positive cannot reset it. This is also why the decision lives on
ESP-2: that flicker-free view of the room already exists there, and the PIR is
wired to the same board, so all of the policy sits in one place while ESP-1
stays a straightforward sensor.

Recent motion also blocks the detector from standing down, independently of what
the camera reports. This covers the case where somebody is in the room but is not
inside any of the nine mapped zones — standing at the edge, or in a corner the
grid does not cover. The camera sees an empty floor and the empty-frame count
climbs, but every movement the PIR picks up pushes the block forward, so the
detector keeps running. In effect the sensor gets to overrule the camera.

Run/idle state is **re-broadcast every two seconds** rather than sent once.
Wireless broadcasts are unacknowledged, so a single lost "stand down" message
would leave the detector running indefinitely; repeating the current state means
any lost message is corrected by the next one. As a further safeguard, ESP-1
resumes detecting on its own if it hears nothing from ESP-2 for 30 seconds, so a
failed controller cannot leave the room unmonitored.

## Wiring

### ESP-1 — OV3660 camera

| Signal | GPIO | Signal | GPIO |
|--------|------|--------|------|
| XCLK | 15 | Y2 | 11 |
| SIOD | 4 | Y3 | 9 |
| SIOC | 5 | Y4 | 8 |
| VSYNC | 6 | Y5 | 10 |
| HREF | 7 | Y6 | 12 |
| PCLK | 13 | Y7 | 18 |
| Y8 | 17 | Y9 | 16 |

### ESP-2 — relays and sensor

| Zone | Appliance | GPIO | Relay type |
|------|-----------|------|------------|
| 0 | Bulb 1 | 26 | active-LOW |
| 1 | Fan 1 | 13 | active-LOW |
| 2 | Bulb 2 | 25 | active-LOW |
| 3 | Fan 2 | 23 | active-LOW |
| 4 | Bulb 3 | 33 | active-LOW |
| 5 | Fan 3 | 14 | active-LOW |
| 6 | Bulb 4 | 32 | active-LOW |
| 7 | Fan 4 | 27 | active-LOW |
| 8 | Bulb 5 | 4 | active-HIGH |
| — | PIR output | 35 | input only |

Zones are numbered row-major, so zone 0 is the top-left of the camera's view:

```
0 1 2
3 4 5
6 7 8
```

Lights and fans alternate in a checkerboard, matching how they are mounted on
the ceiling:

```
bulb   fan    bulb
fan    bulb   fan
bulb   fan    bulb
```

The pattern needs five lights and four fans, which is exactly the hardware. Each
zone still drives its own appliance, so an occupied corner switches on the light
above it without also starting the fan beside it.

**GPIO 12 is deliberately unused.** On this chip it is a boot strapping pin that
selects the flash voltage, and a relay input holding it high prevents the board
from starting.

The relay board should have its own 5 V supply rather than drawing from the
ESP32's regulator — nine coils together draw close to an amp. Grounds must be
common between the two supplies and the board.

## Message format

Both boards broadcast on a fixed channel, so neither needs the other's MAC
address and no router is involved. Messages are distinguished by a leading tag
byte, defined in `occupancy_link.h`:

```c
grid_msg_t   // ESP-1 → ESP-2: the nine zone results, plus sequence number
ctrl_msg_t   // ESP-2 → ESP-1: whether the detector should be running
```

The header is duplicated in both project folders because the boards are built
with different toolchains. The two copies must stay identical.

## Running the project

### What you need

Hardware:

- ESP32-S3 development board with PSRAM (N16R8) and an OV3660 camera module
- ESP32 NodeMCU, 30-pin
- 9-channel relay board
- HC-SR501 PIR motion sensor
- 4 fans and 5 bulbs
- Two 5 V supplies rated at 1 A or more, one for the boards and one for the relays

Software:

- [PlatformIO](https://platformio.org/), for the camera board
- Arduino IDE with the ESP32 board package installed, for the controller
  (*Tools → Board → Boards Manager*, search `esp32`, install the Espressif entry)

There is no training or model-conversion step. The trained network is already
embedded in `esp1_detector/src/model.cc`, so the two boards are all that has to
be built.

The two boards use different tools. The camera board needs the TensorFlow Lite
library, a custom partition layout and PSRAM enabled, which PlatformIO handles
from a single config file; the controller is a plain sketch, so the Arduino IDE
is simpler for it. Upload one board at a time, so there is no doubt about which
serial port belongs to which.

### 1. Flash the camera board (PlatformIO)

1. Install [VS Code](https://code.visualstudio.com/), then add the **PlatformIO
   IDE** extension from the Extensions panel.
2. *File → Open Folder* and choose the **`esp1_detector`** folder. Open that
   folder itself, not the repository root — PlatformIO expects to see
   `platformio.ini` at the top level.
3. Connect the ESP32-S3 to USB.
4. Click the **→** arrow in the blue bar at the bottom of the window to build
   and upload.

From a terminal instead of the editor, the same thing is:

```
cd esp1_detector
pio run -t upload
```

The first build downloads the compiler and the TensorFlow Lite Micro library and
takes several minutes. Later builds take seconds.

### 2. Flash the controller (Arduino IDE)

1. Install the [Arduino IDE](https://www.arduino.cc/en/software).
2. Add ESP32 support: *Tools → Board → Boards Manager*, search for `esp32`, and
   install the entry published by Espressif Systems. This is a one-off step.
3. Open **`esp2_controller/esp2_controller.ino`**. Keep `occupancy_link.h` in
   that same folder; the IDE only finds headers sitting next to the sketch.
4. *Tools → Board →* **ESP32 Dev Module**.
5. *Tools → Port →* the port that appears when the NodeMCU is plugged in.
6. Click the **→** arrow to upload.

If the upload stalls at `Connecting....`, hold the **BOOT** button, tap **EN**,
then release **BOOT** and upload again. Some boards need this to enter
programming mode.

### 3. Wire everything

Follow the tables in [Wiring](#wiring). The relay board needs its own 5 V
supply; nine coils together draw close to an amp, which is more than the
NodeMCU's regulator will pass. Ground must be common between both supplies and
the board, or the relay inputs have no reference to switch against.

Mount the camera so it looks down over the nine zones, then read the
recalibration note below.

### 4. Power up and check

Open a serial monitor on each board at 115200 baud. The camera board should
report:

```
=== Classroom occupancy detector (ESP-1) ===
Camera ready: OV3660, 800x600 RGB565
Model ready, arena in use: 325324 bytes
ESP-NOW ready on channel 1
grid 000 010 000  occupied 1/9  infer 201 ms  seq 1
```

and the controller should pick those up within a second or two:

```
=== Classroom occupancy controller (ESP-2) ===
MAC 68:09:47:52:38:AC, listening on channel 1
Link up: receiving from ESP-1
seq 12    raw 000 010 000 (1/9)   verdict 000 000 000 (0/9)
```

Nothing switches immediately, and that is expected: a zone has to be occupied in
12 of the last 20 frames before its appliance turns on, which takes roughly nine
seconds of someone actually standing there. Sending `s` to the controller prints
a full table of votes, verdicts and appliance states.

Once both boards are behaving, they can run from their own supplies with no
computer attached.

### Recalibrating after moving the camera

`HINV` in `esp1_detector/src/main.cpp` is the perspective transform from the
camera's view to the flat 96×96 grid, and it was measured from the four floor
corners of one particular mounting position. Move or re-aim the camera and that
transform no longer matches, so the model will be looking at the wrong part of
the picture and the zones will not line up with the room.

Re-measuring means photographing the room from the new position, noting the
pixel coordinates of the four corners of the monitored floor area, and computing
the inverse homography from those four point pairs.

### If something does not work

| Symptom | Likely cause |
|---------|--------------|
| Controller never prints `Link up` | The boards are on different channels, or the camera board is not powered |
| Appliances never switch on | Often correct — an empty room keeps everything off by design. Check the `verdict` line rather than the relays |
| Controller will not flash, `Failed to communicate with the flash chip` | Something is connected to GPIO 12, which must stay free |
| Camera board works on USB but not on a power adapter | The supply cannot deliver enough current; peaks go well past 500 mA |
| Zones respond, but the wrong ones | The camera has moved relative to the calibration — see above |

## Notes

The model buffer in `model.cc` is declared `alignas(16)`. This is required, not
cosmetic — the optimised SIMD kernels read weights directly from that buffer and
assume 16-byte alignment. Without it the convolutions return saturated nonsense
at normal speed, with no crash and no error message.
