// Wire format for the ESP-NOW link between the two boards.
//
// Both boards broadcast to FF:FF:FF:FF:FF:FF on a fixed channel, so neither
// needs to know the other's MAC address and no router or access point is
// involved. Each board therefore sees every frame the other sends, and tells
// them apart by the leading tag byte.
//
// This file is duplicated in esp1_detector/include/ and esp2_controller/
// because the two boards are built with different toolchains (PlatformIO and
// the Arduino IDE, which only looks for headers next to the sketch). The two
// copies must stay byte-for-byte identical -- if they drift, the length and
// tag checks on each side will simply drop every packet.

#ifndef OCCUPANCY_LINK_H
#define OCCUPANCY_LINK_H

#include <stdint.h>

// Both radios must sit on this channel. ESP-NOW does not hop channels, so a
// mismatch here looks exactly like a dead link.
static const uint8_t ESPNOW_CHANNEL = 1;

// Leading byte of every message, so a stray packet from some other ESP-NOW
// device on the same channel cannot be mistaken for ours.
static const uint8_t MSG_GRID = 0xA1;   // detector -> controller
static const uint8_t MSG_CTRL = 0xA2;   // controller -> detector

// Detector -> controller: one raw, unsmoothed inference result.
// The grid is row-major, so cell index = row * 3 + col:
//     0 1 2
//     3 4 5
//     6 7 8
typedef struct __attribute__((packed)) {
  uint8_t  tag;         // MSG_GRID
  uint8_t  grid[9];     // per-cell occupancy, 0 or 1
  uint8_t  occupied;    // number of 1s in grid[], saves the receiver counting
  uint16_t infer_ms;    // inference time for this frame, useful as a health signal
  uint32_t seq;         // increments every frame, lets the receiver spot losses
} grid_msg_t;

// Controller -> detector: whether the detector should be running at all.
//
// This is sent as repeated state rather than one-shot commands. ESP-NOW
// broadcasts are unacknowledged, so a single "go to sleep" packet that got
// lost would leave the detector running forever. Re-broadcasting the current
// desired state every couple of seconds means any lost packet is corrected by
// the next one.
typedef struct __attribute__((packed)) {
  uint8_t  tag;         // MSG_CTRL
  uint8_t  run;         // 1 = keep detecting, 0 = go idle
  uint32_t seq;
} ctrl_msg_t;

#endif // OCCUPANCY_LINK_H
