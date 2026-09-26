# Classroom Occupancy Automation

Two ESP32 boards that switch a classroom's fans and lights based on where people
actually are, rather than on a single switch for the whole room.

A ceiling-mounted camera board runs a small neural network that divides the floor
into a 3×3 grid and decides which zones are occupied. A second board receives
those readings over a wireless link, smooths them, and drives one relay per zone.
A PIR motion sensor wakes the camera when someone comes in, and the system goes
back to sleep once the room has been empty for a while.

```
        ┌──────────────────────────┐                  ┌───────────────────────────┐
        │  ESP-1  ESP32-S3 + OV3660│   grid, ~1.4 Hz  │  ESP-2  ESP32 NodeMCU     │
        │                          │ ───────────────► │                           │
        │  capture → warp → CLAHE  │   hello at boot  │  smoothing + thresholds   │
        │  → int8 CNN → 3×3 grid   │ ───────────────► │  relay control            │
        │                          │ ◄─────────────── │  run / sleep decision     │
        └──────────────────────────┘   run / idle     └───────────┬───────────────┘
                                                                  │
                                        ┌─────────────────────────┼──────────────────┐
                                        │                         │                  │
                                  HC-SR501 PIR            4 fans + 5 bulbs      status LED
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

### When the detector runs

The camera does not run all the time. ESP-2 decides when ESP-1 should work:

| Rule | Value | Meaning |
|------|-------|---------|
| At power-up or reset | asleep | the room is assumed empty until the PIR says otherwise |
| Wakes on | any PIR motion | once the sensor has warmed up |
| PIR warm-up | 10 s after ESP-2 powers up | motion is ignored while the HC-SR501 settles |
| Goes back to sleep after | 150 consecutive empty frames | roughly 1.8 minutes of an empty room |
| Motion blocks sleep for | 60 s after the last motion | see below |

While asleep, ESP-1 keeps its camera and radio initialised, so waking up is
immediate, but it captures no frames, runs no inference and sends nothing.

The empty-room count uses the **smoothed** verdict, not the raw frames, so a
single false positive cannot reset it. This is also why the decision lives on
ESP-2: that flicker-free view of the room already exists there, and the PIR is
wired to the same board, so all of the policy sits in one place while ESP-1
stays a straightforward sensor.

Recent motion also blocks the detector from going to sleep, independently of
what the camera reports. This covers the case where somebody is in the room but
is not inside any of the nine mapped zones — standing at the edge, or in a corner
the grid does not cover. The camera sees an empty floor and the empty-frame count
climbs, but the block lasts for as long as the PIR keeps reporting motion plus
60 s afterwards, so the detector keeps running. In effect the sensor gets to
overrule the camera. (The HC-SR501 holds its output high for as long as motion
continues, so the block is refreshed for that whole time, not just when motion
first starts.)

The 10 s warm-up is deliberately short. The PIR only **wakes the camera** — it
never switches an appliance itself, because the relays always follow the
camera's 12-of-20 vote. So the worst a false trigger can do while the sensor is
still settling is run the camera for a couple of minutes in an empty room. If
you see motion reported straight after power-up with nobody around, raise
`PIR_WARMUP_MS` in the sketch to 20–30 s.

### Starting fresh after a power-up or reset

Powering up or resetting **either** board puts the whole system back to the same
clean state: room assumed empty, camera asleep, every appliance off, status LED
off. Only PIR motion starts it again.

ESP-2 does this simply by booting asleep and telling ESP-1 so straight away. The
other direction needs a message, because when only ESP-1 restarts, ESP-2 has no
way to know. So ESP-1 announces itself: on boot it sends a **hello** once a
second until ESP-2 replies with "idle", and ESP-2 treats a hello as "the camera
just restarted" and resets itself to the clean state.

Until that reply arrives, ESP-1 **ignores any "run" command**. Without that, a
restarted ESP-1 could pick up a "run" that ESP-2 was still repeating from before
the restart and wake straight back up, skipping the fresh start.

### Keeping the two boards in step

Run/idle state is **re-broadcast every two seconds** rather than sent once.
Wireless broadcasts are unacknowledged, so a single lost "sleep" message would
leave the detector running indefinitely; repeating the current state means any
lost message is corrected by the next one.

If ESP-2 goes quiet for 10 seconds while ESP-1 is running, ESP-1 goes to sleep on
its own. The relays are on ESP-2, so running the camera without it achieves
nothing — and a silent ESP-2 has most likely been switched off, which should
leave the system asleep rather than busy. In the other direction, if ESP-1 stops
sending frames for 15 seconds while it is supposed to be running, ESP-2 switches
every appliance off.

### Status LED

The LED on ESP-2 shows whether the camera is **actually awake**, as reported by
the camera itself. Each frame ESP-1 sends is proof that it has just captured an
image and run the model, so the LED is driven by those frames arriving rather
than by what ESP-2 has asked for.

| LED | When |
|-----|------|
| Off | at power-up or reset, and whenever the camera is asleep |
| On | as soon as the first frame arrives after a wake-up |
| Off again | immediately when the system goes to sleep, or 3 s after frames stop if ESP-1 fails while it should be running |

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

### ESP-2 — relays, sensor and LED

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
| — | Status LED | 19 | active-HIGH, through a 220–330 Ω resistor to GND |

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

### The PIR sensor

- **VCC goes to 5 V (the `VIN` pin), not 3V3.** The HC-SR501 needs at least
  4.5 V; on 3.3 V it typically never triggers. Its output only swings to 3.3 V,
  so it connects straight to GPIO 35 without a level shifter.
- **Wire it by the printed labels, not by position.** The pin order differs
  between batches; the labels are under the white dome.
- **Fit the trigger jumper on H** (repeat trigger). With no jumper cap the
  trigger mode is undefined and the output may never switch.
- Set the time-delay knob (Tx) fully anticlockwise — ESP-2 already holds for
  60 s after motion — and start with the sensitivity knob (Sx) in the middle.
- Keep it connected whenever ESP-2 is running. GPIO 35 has no internal pull
  resistor, so an unconnected pin floats and can report phantom motion.

## Message format

Both boards broadcast on a fixed channel, so neither needs the other's MAC
address and no router is involved. Messages are distinguished by a leading tag
byte, defined in `occupancy_link.h`:

```c
grid_msg_t    // ESP-1 → ESP-2: the nine zone results, plus sequence number
ctrl_msg_t    // ESP-2 → ESP-1: whether the detector should be running
hello_msg_t   // ESP-1 → ESP-2: "I have just restarted" (sent at boot)
```

Every receiver checks both the tag and the length, so a stray packet from some
other ESP-NOW device on the same channel cannot be mistaken for one of these.

The header is duplicated in both project folders because the boards are built
with different toolchains. The two copies must stay identical, and both boards
must be flashed from the same version.

## Running the project

### What you need

Hardware:

- ESP32-S3 development board with PSRAM (N16R8) and an OV3660 camera module
- ESP32 NodeMCU, 30-pin
- 9-channel relay board
- HC-SR501 PIR motion sensor
- An LED and a 220–330 Ω resistor
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

Open a serial monitor on each board at 115200 baud. After some memory and model
details, the camera board should report that its radio is up and that it is
waiting, asleep:

```
model_tflite @ 0x3c0a1ae0  (16-byte aligned: yes)
Model ready (input int8). arena_used=325324 bytes.
set_channel(1) -> ESP_OK | radio is ACTUALLY on channel 1
sizeof(grid_msg_t) = 17, ctrl_msg_t = 6, hello_msg_t = 5 bytes (expect 17/6/5)
esp_now_init() -> ESP_OK
esp_now_add_peer(broadcast) -> ESP_OK
esp_now_register_recv_cb() -> ESP_OK
Sender MAC: XX:XX:XX:XX:XX:XX
ESP-NOW ready on channel 1
FRESH START: idle -- no capture, no grids -- until ESP-2 sends RUN (PIR motion).
Sending HELLO to ESP-2 every 1 s until it acknowledges with IDLE ...
>>> SYNCED with ESP-2 after 1 HELLO(s) -- system reset to fresh idle
```

The address on the first line varies between builds; what matters is
`aligned: yes`. The controller should report the same channel and the same three
message sizes, then:

```
FRESH START: room assumed EMPTY, ESP-1 told to SLEEP, all appliances OFF, LED OFF.
Only PIR motion wakes ESP-1.
PIR warming up: motion is ignored for the first 10 s.
```

Ten seconds later it prints `PIR ARMED`. Walk across the PIR's field of view and
you should see, in order:

- on the controller, a `PIR: MOTION DETECTED` banner and `DETECTOR ON`
- on the camera board, `>>> RUNNING: ESP-2 woke the detector`, followed by a
  timing line and a 3×3 grid for every frame
- on the controller, `>>> ESP-1 AWAKE: its frames are arriving -- status LED ON`,
  with the LED lighting up

Nothing switches immediately, and that is expected: a zone has to be occupied in
12 of the last 20 frames before its appliance turns on, which takes roughly nine
seconds of someone actually standing there. Leave the room and, about two minutes
later, the controller prints `DETECTOR OFF`, the LED goes out, and the camera
board prints `>>> IDLE`.

Once both boards are behaving, they can run from their own supplies with no
computer attached.

### Serial commands

On the controller (ESP-2), with the line ending set to *Newline*:

| Command | Effect |
|---------|--------|
| `s` | Print the PIR state, run/sleep state, empty-frame count, and a table of votes, verdicts and appliance states |
| `v` | Switch between the full per-frame printout and one compact line per frame (easier to follow; still shows the PIR and LED state) |

On the camera board (ESP-1), for debugging the image pipeline while it is awake:

| Command | Effect |
|---------|--------|
| `B` | Dump the exact 96×96 image the model sees, as hex |
| `D` | Dump the same image as raw bytes |

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
| Nothing happens when you walk in | The system starts asleep and only the PIR wakes it. Wait for `PIR ARMED`, then move *across* the sensor's view — see [The PIR sensor](#the-pir-sensor) |
| PIR never reports motion | VCC on 3V3 instead of 5 V, the trigger jumper missing, or wires on the wrong pins. Test it alone: OUT → 330 Ω → LED → GND should light when you move |
| Motion reported straight after power-up with nobody around | The PIR is still settling — raise `PIR_WARMUP_MS` in the sketch |
| Controller never prints `LINK UP` after a wake-up | The boards are on different channels, or the camera board is not powered |
| Status LED never lights | It only lights while the camera is awake *and* sending frames; check the camera board woke up and is powered |
| Appliances never switch on | Often correct — an empty room keeps everything off by design. Check the verdict in the printout rather than the relays |
| Controller will not flash, `Failed to communicate with the flash chip` | Something is connected to GPIO 12, which must stay free |
| Camera board works on USB but not on a power adapter | The supply cannot deliver enough current; peaks go well past 500 mA |
| Controller prints `Receiver MAC: 00:00:00:00:00:00` | The radio did not start; reset the board |
| Every zone reads occupied, all the time | The model buffer has lost its alignment — see the note below |
| Zones respond, but the wrong ones | The camera has moved relative to the calibration — see above |

## Notes

The model buffer in `model.cc` is declared `alignas(16)`. This is required, not
cosmetic — the optimised SIMD kernels read weights directly from that buffer and
assume 16-byte alignment. Without it the convolutions return saturated nonsense
at normal speed, with no crash and no error message: every zone reads about
0.996 and the whole grid shows occupied. Where the buffer lands depends on
everything else linked into the program, so an unaligned buffer can work by luck
and then break when unrelated code is added. The camera board prints the
buffer's address and alignment at boot for this reason.

A few other details were needed to make the boards reliable on their own
supplies, and are easy to undo by accident:

- **The camera board never waits for a serial connection.** Its USB serial port
  only reports "connected" when a computer opens it, so waiting for that would
  hang the board forever on a plain power adapter. It waits at most 1.5 s.
- **The controller turns WiFi power saving off.** In its default mode the radio
  sleeps between beacons, and an ESP-NOW receiver that is asleep simply misses
  most incoming frames.
- **Both boards wait for the radio to be ready before configuring it.** The
  Arduino core starts WiFi in the background; setting the channel too early fails
  without an error, and the board then listens on the wrong channel. Each board
  reads the channel back and prints it at boot.
