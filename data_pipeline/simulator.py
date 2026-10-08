#!/usr/bin/env python3
"""
LogiEdge -- Cold-Chain Truck Sensor Simulator  (Task C1)

Generates three synchronised sensor streams for a refrigerated truck and
publishes them to a local Mosquitto broker and/or a JSONL file.

Streams
-------
  temperature    1.0 Hz   N(4.0 C, 0.3)      setpoint 4 C
  vibration_rms  0.5 Hz   N(0.45 g, 0.05)    compressor RMS
  door_event     discrete OPEN / CLOSE with timestamp

Anomaly modes (--anomaly)
-------------------------
  none        all streams nominal                             -> Class 0 Normal
  temp_drift  temperature ramps +0.08 C per reading            -> Class 1 Warning
  vibration   vibration steps to N(1.2 g, 0.15) bearing wear   -> (diagnostic)
  combined    temperature drift AND vibration step together    -> Class 2 Critical

DESIGN NOTE -- bounded sawtooth drift
-------------------------------------
The specified drift rate of +0.08 C per reading at 1 Hz is +4.8 C/min. Left
unbounded for the 15-minute Warning capture required by Task D1 the cargo would
reach +72 C, and every window after the first minute would satisfy the Class 2
definition (>3 C outside setpoint) rather than Class 1 (1-3 C outside setpoint).

We therefore bound the drift with a recovery cycle: the offset ramps at the
specified rate until it reaches a mode-specific ceiling, then the compressor
"catches up" and pulls the cargo back toward setpoint, and the cycle repeats.
This is the physical behaviour of a refrigeration unit that is struggling but
not yet dead, it preserves the specified +0.08 C/reading ramp rate exactly, and
it produces a Warning capture whose windows genuinely sit in the 1-3 C band.

  temp_drift : ceiling +3.0 C  -> windows land in the Class 1 band
  combined   : ceiling +8.0 C  -> sustained breach, Class 2 band

Usage
-----
  python simulator.py --anomaly none --duration 1200 --speed 60
  python simulator.py --anomaly combined --duration 900 --sink both \
                      --file ../training/data/critical.jsonl
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Simulation constants -- single source of truth, imported by other modules
# --------------------------------------------------------------------------
TEMP_HZ = 1.0  # temperature sample rate
VIB_HZ = 0.5  # vibration RMS sample rate
TEMP_SETPOINT = 4.0  # C
TEMP_SIGMA = 0.3  # C, sensor + cabinet noise
VIB_NORMAL_MEAN = 0.45  # g
VIB_NORMAL_SIGMA = 0.05  # g
VIB_FAULT_MEAN = 1.20  # g, bearing wear signature
VIB_FAULT_SIGMA = 0.15  # g
DRIFT_PER_READING = 0.08  # C per temperature reading, as specified
RECOVERY_PER_READING = 0.25  # C per reading, compressor catch-up
FAULT_ONSET_SECONDS = 90.0  # mechanical degradation ramp, see note below
DRIFT_CEILING = {"temp_drift": 3.0, "combined": 8.0}  # C above setpoint
DRIFT_FLOOR = 0.3  # C, where recovery hands back to ramp
DOOR_OPEN_MEAN_INTERVAL = 240.0  # s between door openings (exponential)
DOOR_OPEN_DURATION = (20.0, 90.0)  # s, uniform
DOOR_TEMP_RISE = 0.020  # C per second while the door is open
DOOR_TEMP_CAP = 0.80  # C, saturation of the door-induced rise
DOOR_TEMP_DECAY = 0.030  # C per second of compressor recovery after closing

TOPIC_ROOT = "logibridge/trucks"
QOS_TELEMETRY = 0  # superseded by the next sample; loss is tolerable
QOS_EVENT = 1  # must arrive; duplicates are idempotent by timestamp

MODES = ("none", "temp_drift", "vibration", "combined")


# --------------------------------------------------------------------------
# Sensor models
# --------------------------------------------------------------------------
class TemperatureSensor:
    """Cargo compartment probe with optional bounded drift and door coupling."""

    def __init__(self, mode, rng, onset=FAULT_ONSET_SECONDS):
        self.rng = rng
        self.drifting = mode in ("temp_drift", "combined")
        self.ceiling = DRIFT_CEILING.get(mode, 0.0)
        self.onset = max(1e-6, onset)
        self.offset = 0.0  # C above setpoint, the fault state
        self.ramping = True  # True = fault growing, False = recovering
        self.door_offset = 0.0  # C contributed by an open door

    def step(self, dt, door_open, sim_t):
        # The refrigeration fault develops over FAULT_ONSET_SECONDS rather than
        # appearing at full strength, for the reason given in VibrationSensor.
        # DRIFT_PER_READING remains the fully-developed rate the brief specifies.
        severity = min(1.0, sim_t / self.onset) if self.drifting else 0.0

        if self.drifting:
            if self.ramping:
                self.offset += DRIFT_PER_READING * severity
                if self.offset >= self.ceiling:
                    self.ramping = False
            else:
                self.offset -= RECOVERY_PER_READING
                if self.offset <= DRIFT_FLOOR:
                    self.offset = DRIFT_FLOOR
                    self.ramping = True

        # An open door adds load, saturating at DOOR_TEMP_CAP, and decays once
        # the door shuts. The cap is deliberately below the +/-1 C half-width of
        # the Class 0 definition so that routine loading stops at a door does
        # not relabel Normal data as Warning.
        if door_open:
            self.door_offset = min(
                DOOR_TEMP_CAP, self.door_offset + DOOR_TEMP_RISE * dt
            )
        else:
            self.door_offset = max(0.0, self.door_offset - DOOR_TEMP_DECAY * dt)

        noise = self.rng.gauss(0.0, TEMP_SIGMA)
        return TEMP_SETPOINT + self.offset + self.door_offset + noise


class VibrationSensor:
    """Compressor vibration RMS with a gradual bearing-degradation onset.

    DESIGN NOTE -- why the fault ramps in rather than stepping
    ----------------------------------------------------------
    A bearing does not jump from 0.45 g to 1.2 g in one sample; it degrades.
    Modelling the fault as an instantaneous step makes the three classes
    linearly separable with zero overlap, and the classifier then scores 100%
    -- a number that says more about the simulator than about the model.

    Ramping the fault amplitude in over FAULT_ONSET_SECONDS reproduces the
    physically real situation: for the first minute and a half of a developing
    fault the sensor signature genuinely resembles normal operation, so the
    windows in that interval are honestly ambiguous. Residual validation error
    concentrates there, which is where a real cold-chain monitor is also
    uncertain, and it is the interval the 90-second detection SLA is written
    against.
    """

    def __init__(self, mode, rng, onset=FAULT_ONSET_SECONDS):
        self.rng = rng
        self.faulty = mode in ("vibration", "combined")
        self.onset = max(1e-6, onset)

    def step(self, sim_t):
        severity = min(1.0, sim_t / self.onset) if self.faulty else 0.0
        mean = VIB_NORMAL_MEAN + severity * (VIB_FAULT_MEAN - VIB_NORMAL_MEAN)
        sigma = VIB_NORMAL_SIGMA + severity * (VIB_FAULT_SIGMA - VIB_NORMAL_SIGMA)
        value = self.rng.gauss(mean, sigma)
        # Bearing wear is impulsive: occasional impacts lift the kurtosis
        # feature, and they too become more frequent as the fault develops.
        if severity > 0 and self.rng.random() < 0.08 * severity:
            value += abs(self.rng.gauss(0.0, 0.45 * severity))
        return max(0.0, value)


class DoorModel:
    """Poisson-arrival door openings of uniform duration."""

    def __init__(self, rng):
        self.rng = rng
        self.open = False
        self.next_change = rng.expovariate(1.0 / DOOR_OPEN_MEAN_INTERVAL)

    def step(self, sim_t):
        """Return 'OPEN', 'CLOSE' or None for this instant."""
        if sim_t < self.next_change:
            return None
        if self.open:
            self.open = False
            self.next_change = sim_t + self.rng.expovariate(
                1.0 / DOOR_OPEN_MEAN_INTERVAL
            )
            return "CLOSE"
        self.open = True
        self.next_change = sim_t + self.rng.uniform(*DOOR_OPEN_DURATION)
        return "OPEN"


# --------------------------------------------------------------------------
# Output sinks
# --------------------------------------------------------------------------
class Sink:
    """Fan-out to an MQTT broker and/or a JSONL file."""

    def __init__(self, use_mqtt, path, host, port, truck_id):
        self.client = None
        self.fh = None
        self.truck_id = truck_id
        self.count = 0

        if use_mqtt:
            import paho.mqtt.client as mqtt

            # CallbackAPIVersion is required by paho-mqtt 2.x and absent in 1.x.
            try:
                self.client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"logibridge-sim-{truck_id}",
                )
            except AttributeError:
                self.client = mqtt.Client(client_id=f"logibridge-sim-{truck_id}")
            self.client.connect(host, port, keepalive=60)
            self.client.loop_start()
            print(f"[sim] connected to MQTT broker {host}:{port}", file=sys.stderr)

        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.fh = open(path, "w")
            print(f"[sim] writing JSONL to {path}", file=sys.stderr)

    def publish(self, subtopic, payload, qos):
        line = json.dumps(payload)
        if self.client is not None:
            topic = f"{TOPIC_ROOT}/{self.truck_id}/{subtopic}"
            self.client.publish(topic, line, qos=qos)
        if self.fh is not None:
            self.fh.write(line + "\n")
        self.count += 1

    def close(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
        if self.fh is not None:
            self.fh.close()


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run(args):
    rng = random.Random(args.seed)
    temp = TemperatureSensor(args.anomaly, rng, args.onset)
    vib = VibrationSensor(args.anomaly, rng, args.onset)
    door = DoorModel(rng)

    sink = Sink(
        use_mqtt=args.sink in ("mqtt", "both"),
        path=args.file if args.sink in ("file", "both") else None,
        host=args.host,
        port=args.port,
        truck_id=args.truck_id,
    )

    tick = 1.0 / TEMP_HZ  # 1.0 s base tick
    vib_every = int(round(TEMP_HZ / VIB_HZ))  # vibration on every 2nd tick
    n_ticks = int(args.duration * TEMP_HZ)
    wall_per_tick = tick / args.speed

    print(
        f"[sim] truck={args.truck_id} mode={args.anomaly} "
        f"duration={args.duration}s speed={args.speed}x "
        f"(~{args.duration / args.speed:.1f}s wall clock)",
        file=sys.stderr,
    )

    t0 = time.time()
    try:
        for i in range(n_ticks):
            sim_t = i * tick
            wall_ts = t0 + sim_t / args.speed

            # --- door events are evaluated first; they feed the temperature model
            event = door.step(sim_t)
            if event is not None:
                sink.publish(
                    "events/door",
                    {
                        "truck_id": args.truck_id,
                        "ts": wall_ts,
                        "sim_t": sim_t,
                        "stream": "door_event",
                        "value": event,
                        "unit": "state",
                        "mode": args.anomaly,
                    },
                    QOS_EVENT,
                )

            # --- temperature at 1 Hz
            sink.publish(
                "sensors/temperature",
                {
                    "truck_id": args.truck_id,
                    "ts": wall_ts,
                    "sim_t": sim_t,
                    "stream": "temperature",
                    "value": round(temp.step(tick, door.open, sim_t), 4),
                    "unit": "C",
                    "mode": args.anomaly,
                },
                QOS_TELEMETRY,
            )

            # --- vibration at 0.5 Hz
            if i % vib_every == 0:
                sink.publish(
                    "sensors/vibration",
                    {
                        "truck_id": args.truck_id,
                        "ts": wall_ts,
                        "sim_t": sim_t,
                        "stream": "vibration_rms",
                        "value": round(vib.step(sim_t), 4),
                        "unit": "g",
                        "mode": args.anomaly,
                    },
                    QOS_TELEMETRY,
                )

            if wall_per_tick > 0:
                time.sleep(wall_per_tick)

    except KeyboardInterrupt:
        print("\n[sim] interrupted", file=sys.stderr)
    finally:
        sink.close()
        print(
            f"[sim] published {sink.count} messages in "
            f"{time.time() - t0:.1f}s wall clock",
            file=sys.stderr,
        )


def main():
    p = argparse.ArgumentParser(description="LogiEdge cold-chain sensor simulator")
    p.add_argument(
        "--anomaly",
        choices=MODES,
        default="none",
        help="fault mode to inject (default: none)",
    )
    p.add_argument(
        "--duration", type=float, default=1200, help="simulated seconds (default 1200)"
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="wall-clock acceleration; 60 means 60x faster than real time",
    )
    p.add_argument("--truck-id", default="TRK-001")
    p.add_argument(
        "--sink",
        choices=("mqtt", "file", "both"),
        default="mqtt",
        help="where to send samples (default: mqtt)",
    )
    p.add_argument("--file", default=None, help="JSONL output path for file sinks")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument(
        "--onset",
        type=float,
        default=FAULT_ONSET_SECONDS,
        help="seconds over which an injected fault ramps to full severity",
    )
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.sink in ("file", "both") and not args.file:
        p.error("--sink file/both requires --file")
    if args.speed <= 0:
        p.error("--speed must be positive")

    run(args)


if __name__ == "__main__":
    main()
