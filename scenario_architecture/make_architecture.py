#!/usr/bin/env python3
"""
LogiEdge -- System Architecture Diagram  (Task A2)

Renders system_architecture.png: the truck edge node, its sensors, the local
MQTT broker, the inference pipeline, the local alert log, the cellular uplink
and the operations centre backend.

The diagram is drawn in code rather than a drawing tool so it regenerates
from one source and cannot drift from the system it documents.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
GREY = "#8a8880"


def box(ax, x, y, w, h, label, sub=None, color=BLUE, fill="#ffffff", fs=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.7, edgecolor=color, facecolor=fill, zorder=3,
        )
    )
    ty = y + h / 2 + (0.022 if sub else 0)
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=fs, color=INK, fontweight="semibold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.032, sub, ha="center", va="center",
                fontsize=7.6, color=MUTED, zorder=4)


def arrow(ax, p0, p1, color=GREY, style="-|>", lw=1.5, ls="-", label=None,
          lx=None, ly=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=13,
            linewidth=lw, color=color, linestyle=ls, zorder=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(lx if lx is not None else (p0[0] + p1[0]) / 2,
                ly if ly is not None else (p0[1] + p1[1]) / 2,
                label, ha="center", va="center", fontsize=7.3, color=MUTED,
                zorder=5,
                path_effects=[pe.withStroke(linewidth=3.2, foreground=SURFACE)])


def main():
    fig, ax = plt.subplots(figsize=(14.5, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # ---------------------------------------------------------- truck boundary
    ax.add_patch(Rectangle((0.015, 0.075), 0.635, 0.845, linewidth=2.0,
                           edgecolor=BLUE, facecolor="#f4f8fd",
                           linestyle=(0, (7, 4)), zorder=0))
    ax.text(0.033, 0.885, "REFRIGERATED TRUCK  —  edge node (Raspberry Pi 5 + AI HAT+, 7.5 W)",
            fontsize=10.5, color=BLUE, fontweight="bold", zorder=5)
    ax.text(0.033, 0.858, "everything inside this boundary runs with no cellular dependency",
            fontsize=8.2, color=MUTED, style="italic", zorder=5)

    # ------------------------------------------------------------- sensors
    ax.text(0.085, 0.805, "SENSORS", fontsize=8.6, color=MUTED, fontweight="bold")
    box(ax, 0.035, 0.690, 0.155, 0.085, "Temperature", "PT100 · 1 Hz · setpoint 4 °C",
        color=AQUA, fs=9)
    box(ax, 0.035, 0.575, 0.155, 0.085, "Vibration", "MEMS 500 Hz ×3 → RMS 0.5 Hz",
        color=AQUA, fs=9)
    box(ax, 0.035, 0.460, 0.155, 0.085, "Door switch", "discrete OPEN / CLOSE",
        color=AQUA, fs=9)

    # --------------------------------------------------------- local broker
    box(ax, 0.247, 0.560, 0.150, 0.215, "Local MQTT\nbroker", "Mosquitto · localhost:1883",
        color=ORANGE, fill="#fef6f2", fs=10)
    ax.text(0.322, 0.533, "logibridge/trucks/{id}/…", fontsize=7.4, color=MUTED,
            ha="center", family="monospace")

    for sy in (0.7325, 0.6175, 0.5025):
        arrow(ax, (0.190, sy), (0.247, 0.6675), color=AQUA, rad=0.08)
    ax.text(0.218, 0.775, "I²C /\nSPI /\nGPIO", fontsize=7.0, color=MUTED,
            ha="center", va="center")

    # ----------------------------------------------------- inference pipeline
    ax.add_patch(Rectangle((0.432, 0.400), 0.196, 0.395, linewidth=1.3,
                           edgecolor=VIOLET, facecolor="#f6f5fc",
                           linestyle=(0, (4, 3)), zorder=1))
    ax.text(0.530, 0.772, "INFERENCE PIPELINE  (Docker)", fontsize=8.4,
            color=VIOLET, fontweight="bold", ha="center", zorder=5)

    box(ax, 0.447, 0.683, 0.166, 0.070, "Preprocess", "5-sample MA · 30 s / 10 s window",
        color=VIOLET, fs=8.8)
    box(ax, 0.447, 0.594, 0.166, 0.070, "6 features + fusion",
        "feature-level concatenation", color=VIOLET, fs=8.8)
    box(ax, 0.447, 0.505, 0.166, 0.070, "Normalise", "frozen training_stats.npy",
        color=VIOLET, fs=8.8)
    box(ax, 0.447, 0.416, 0.166, 0.070, "TFLite INT8 model",
        "MODEL_PATH switches M1/M2/M3", color=VIOLET, fs=8.8)

    arrow(ax, (0.397, 0.6675), (0.447, 0.7180), color=ORANGE, rad=0.0,
          label="subscribe\nQoS 0", lx=0.420, ly=0.772)
    for y0, y1 in ((0.683, 0.664), (0.594, 0.575), (0.505, 0.486)):
        arrow(ax, (0.530, y0), (0.530, y1), color=VIOLET)

    # --------------------------------------------- classification + alert log
    box(ax, 0.247, 0.268, 0.150, 0.085, "Classification",
        "Normal / Warning / Critical", color=ORANGE, fill="#fef6f2", fs=9.2)
    arrow(ax, (0.447, 0.451), (0.397, 0.3105), color=VIOLET, rad=0.15,
          label="publish  .../inference\nQoS 1", lx=0.452, ly=0.352)

    box(ax, 0.035, 0.268, 0.155, 0.085, "Local alert log",
        "append-only JSONL · fsync", color="#c0392b", fill="#fdf3f2", fs=9.2)
    arrow(ax, (0.247, 0.3105), (0.190, 0.3105), color="#c0392b", lw=1.9,
          label="write FIRST\nsynced=false", lx=0.218, ly=0.235)

    box(ax, 0.247, 0.118, 0.150, 0.080, "Driver alert",
        "cab buzzer + lamp", color="#c0392b", fill="#fdf3f2", fs=9.2)
    arrow(ax, (0.322, 0.268), (0.322, 0.198), color="#c0392b", lw=1.9)

    box(ax, 0.447, 0.268, 0.166, 0.085, "PSI drift monitor",
        "rolling 100 · alert > 0.25", color=ORANGE, fill="#fef6f2", fs=9.2)
    arrow(ax, (0.397, 0.3105), (0.447, 0.3105), color=ORANGE, rad=0.0)

    box(ax, 0.447, 0.118, 0.166, 0.080, "Store-and-forward\nuplink queue",
        "replays unsynced records", color=GREY, fill="#f7f7f5", fs=8.8)
    arrow(ax, (0.113, 0.268), (0.447, 0.158), color="#c0392b", ls=(0, (4, 3)),
          rad=-0.10, label="queued while offline", lx=0.268, ly=0.080)

    # ------------------------------------------------------- cellular uplink
    ax.add_patch(FancyBboxPatch(
        (0.672, 0.360), 0.098, 0.230,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8, edgecolor=GREY, facecolor="#f2f2ef",
        linestyle=(0, (5, 3)), zorder=3))
    ax.text(0.721, 0.520, "Cellular\nuplink", ha="center", va="center",
            fontsize=9.6, color=INK, fontweight="semibold", zorder=4)
    ax.text(0.721, 0.455, "M2M SIM · TLS\n₹0.10 / MB", ha="center", va="center",
            fontsize=7.6, color=MUTED, zorder=4)
    ax.text(0.721, 0.392, "INTERMITTENT", ha="center", va="center",
            fontsize=7.4, color="#c0392b", fontweight="bold", zorder=4)

    arrow(ax, (0.613, 0.158), (0.721, 0.360), color=GREY, rad=-0.18,
          label="alerts + heartbeat\nonly  (0.13 MB/day)", lx=0.690, ly=0.225)

    ax.text(0.721, 0.320, "7 dead zones\n35–90 min\nNashik–Aurangabad",
            ha="center", va="top", fontsize=7.2, color="#c0392b", zorder=4)

    # ------------------------------------------------------ operations centre
    ax.add_patch(Rectangle((0.800, 0.115), 0.190, 0.700, linewidth=2.0,
                           edgecolor=VIOLET, facecolor="#f6f5fc",
                           linestyle=(0, (7, 4)), zorder=0))
    ax.text(0.895, 0.785, "OPERATIONS CENTRE", fontsize=10.0, color=VIOLET,
            fontweight="bold", ha="center", zorder=5)

    box(ax, 0.818, 0.660, 0.154, 0.080, "Alert ingest", "acknowledges → synced=true",
        color=VIOLET, fs=9)
    box(ax, 0.818, 0.550, 0.154, 0.080, "Fleet dashboard", "85 trucks live status",
        color=VIOLET, fs=9)
    box(ax, 0.818, 0.440, 0.154, 0.080, "Chain-of-custody\narchive",
        "regulatory evidence", color=VIOLET, fs=9)
    box(ax, 0.818, 0.330, 0.154, 0.080, "Model registry", "local Docker registry",
        color=VIOLET, fs=9)
    box(ax, 0.818, 0.220, 0.154, 0.080, "Ansible controller",
        "logibridge_deploy.yml", color=VIOLET, fs=9)

    arrow(ax, (0.770, 0.490), (0.818, 0.700), color=GREY, rad=0.16)
    arrow(ax, (0.818, 0.260), (0.770, 0.440), color=AQUA, rad=0.16, lw=1.8)
    ax.text(0.793, 0.165, "OTA: canary 10 → 85\n280 KB model layer only",
            ha="center", va="center", fontsize=7.3, color=AQUA, zorder=5,
            path_effects=[pe.withStroke(linewidth=3.2, foreground=SURFACE)])

    # ------------------------------------------------------------- footnote
    fig.text(0.5, 0.030,
             "Detection path:  sensor → local broker → preprocess → inference → local alert log → driver.  "
             "No element of this path crosses the cellular link.\n"
             "The uplink carries decisions outward, never inward — loss of coverage costs visibility at the "
             "operations centre, never detection on the truck.",
             ha="center", fontsize=8.8, color=MUTED)

    fig.suptitle("LogiEdge — Cold-Chain Edge AI System Architecture   |   "
                 "BITS Pilani WILP AIML ZG535   |   Group 18",
                 fontsize=12.0, y=0.965)

    out = os.path.join(HERE, "system_architecture.png")
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
