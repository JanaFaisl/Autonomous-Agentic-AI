"""Generate the paper's two architecture figures as PNGs.

Kept as a script rather than hand-drawn images so the figures cannot drift from
the implementation: if an agent is added, a role renamed, or the quality gate
wired into a revision pass, edit this file and regenerate. The previous
hand-drawn diagrams depicted agents that never existed and feedback edges the
code does not implement, which is exactly the failure this avoids.

    python -m evaluation.make_figures            # writes figures/*.png at 300dpi
    python -m evaluation.make_figures --dpi 600  # print quality

Every element below is traceable to the code:
  * six pipeline agents            agents/project_manager_agent.py DEFAULT_STEPS
  * no delegation, no feedback     allow_delegation=False on all eight agents
  * advisory quality gate          ui/main_ui.py reads gate_decision for display
  * deterministic renderer         utils/code_scaffold.py ("no LLM calls")
  * fallback path                  _direct_api_call in five agent modules
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# Muted palette: prints legibly in greyscale, which many reviewers still use.
C_UI = "#dce9f7"
C_ORCH = "#e4dcf2"
C_CORE = "#d8eef0"
C_GOV = "#fae3cd"
C_REND = "#d9eeda"
C_APP = "#e8e8e8"
C_SIDE = "#f2f2f2"
C_TERM = "#fadadd"
EDGE = "#333333"
MUTED = "#777777"


def box(ax, x, y, w, h, face, *, lw=1.4, ls="solid", ec=EDGE, r=0.05):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=face, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2,
    ))


def arrow(ax, p1, p2, *, ls="solid", color=EDGE, lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, linestyle=ls, color=color, zorder=3,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1,
    ))


def label(ax, x, y, text, *, size=8, weight="normal", color="#111111", style="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight=weight, color=color, style=style, zorder=4)


# ---------------------------------------------------------------------------
# Figure 1 -- layered architecture
# ---------------------------------------------------------------------------

def figure_architecture(dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 100)          # tight to the drawn content; avoids a white band on top
    ax.axis("off")

    LX, LW = 8, 68          # main column
    layers = [
        (98, 11, C_UI, "User Interaction Layer",
         "Streamlit",
         "project idea input  ·  multiple-choice elicitation loop\n"
         "artifact display  ·  session state"),
        (83, 11, C_ORCH, "Orchestration Layer",
         "Project Manager Agent (CrewAI)",
         "determines step order  ·  maintains workflow state\n"
         "persists each artifact as it completes"),
        (66, 13, C_CORE, "Core Development Layer", None,
         "Requirements Analyst  →  Database  →  Design Generation  →  Planning Manager\n\n"
         "each agent consumes its predecessor's schema-validated JSON"),
        (50, 12, C_GOV, "Governance Layer", None,
         "Quality/Process Manager    ·    Support Manager\n\n"
         "quality report is advisory: displayed, not consumed by any agent"),
        (33, 12, C_REND, "Deterministic Renderer", "no LLM calls",
         "schema DDL + per-table REST routes  ·  one React page per screen\n"
         "CSS derived from the generated colour scheme"),
        (18, 10, C_APP, "Runnable Application", None,
         "Express + node:sqlite backend    ·    React / Vite frontend"),
    ]

    centres = []
    for top, h, colour, title, sub, body in layers:
        y = top - h
        box(ax, LX, y, LW, h, colour)
        cx = LX + LW / 2
        ty = top - 2.6
        label(ax, cx, ty, title, size=11, weight="bold")
        if sub:
            label(ax, cx, ty - 2.6, sub, size=8.5, style="italic", color="#444444")
            body_y = y + h / 2 - 2.4
        else:
            body_y = y + h / 2 - 1.6
        label(ax, cx, body_y, body, size=8)
        centres.append((cx, top, y))

    for i in range(len(centres) - 1):
        _, _, y_bottom = centres[i]
        cx, top_next, _ = centres[i + 1]
        arrow(ax, (cx, y_bottom), (cx, top_next))

    # Side annotations
    box(ax, 80, 74, 17, 8, C_SIDE, ls="dashed", lw=1.1, ec=MUTED)
    label(ax, 88.5, 79.4, "Artifact store", size=8.5, weight="bold")
    label(ax, 88.5, 76.4, "session-scoped", size=7.5, color="#555555")
    arrow(ax, (LX + LW, 77.5), (80, 78), ls="dashed", color=MUTED, lw=1.1)

    box(ax, 80, 56, 17, 10, C_SIDE, ls="dashed", lw=1.1, ec=MUTED)
    label(ax, 88.5, 63.4, "Anthropic API", size=8.5, weight="bold")
    label(ax, 88.5, 60.6, "claude-sonnet-4-5", size=7.5, color="#555555")
    label(ax, 88.5, 58.2, "T = 0.2", size=7.5, color="#555555")
    arrow(ax, (LX + LW, 59.5), (80, 61), ls="dashed", color=MUTED, lw=1.1)

    label(ax, 50, 7,
          "Generative non-determinism is confined to the agent layers; all code is emitted deterministically.",
          size=8.5, style="italic", color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "general_architecture.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2 -- agent interaction
# ---------------------------------------------------------------------------

def figure_interaction(dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(17.0, 7.2))
    ax.set_xlim(0, 222)
    ax.set_ylim(2, 92)
    ax.axis("off")

    # Gap must exceed the widest edge label, or labels overlap the boxes.
    ROW_Y, BH, BW, GAP = 46, 13, 17, 7.5
    chain = [
        ("User", C_SIDE, None),
        ("Chat\nAssistant", C_UI, None),
        ("Requirements\nAnalyst", C_CORE, None),
        ("Database", C_CORE, None),
        ("Design\nGeneration", C_CORE, None),
        ("Planning\nManager", C_CORE, None),
        ("Quality/Process\nManager", C_GOV, None),
        ("Support\nManager", C_GOV, None),
        ("Deterministic\nrenderer", C_REND, None),
    ]
    edge_labels = ["", "clarified", "req.\nJSON", "schema\nJSON",
                   "design\nJSON", "plan\nJSON", "artifacts", "all"]

    xs = []
    x = 4.0
    for name, colour, _ in chain:
        box(ax, x, ROW_Y, BW, BH, colour)
        label(ax, x + BW / 2, ROW_Y + BH / 2, name, size=8.5, weight="bold")
        xs.append(x)
        x += BW + GAP

    for i in range(len(chain) - 1):
        x1 = xs[i] + BW
        x2 = xs[i + 1]
        arrow(ax, (x1, ROW_Y + BH / 2), (x2, ROW_Y + BH / 2))
        if edge_labels[i]:
            # Sit labels ABOVE the row: inside the gap they would clip the boxes.
            label(ax, (x1 + x2) / 2, ROW_Y + BH + 3.4, edge_labels[i],
                  size=7.0, color="#444444")

    # Elicitation loop over the Chat Assistant
    cx = xs[1] + BW / 2
    ax.add_patch(FancyArrowPatch(
        (cx - 5, ROW_Y + BH), (cx + 5, ROW_Y + BH),
        arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=EDGE,
        connectionstyle="arc3,rad=-1.4", zorder=3))
    label(ax, cx, ROW_Y + BH + 9.5, "multiple-choice\nelicitation loop", size=6.8, color="#444444")

    # Project Manager -- invocation, NOT delegation
    pm_y, pm_h = 72, 11
    pm_x = xs[2]
    pm_w = xs[7] + BW - pm_x
    box(ax, pm_x, pm_y, pm_w, pm_h, C_ORCH)
    label(ax, pm_x + pm_w / 2, pm_y + pm_h - 4.0, "Project Manager Agent  (Team Leader)",
          size=10, weight="bold")
    label(ax, pm_x + pm_w / 2, pm_y + 3.4,
          "computes step order and invokes agents in sequence — no delegation",
          size=7.8, style="italic", color="#444444")
    for i in (2, 4, 7):
        arrow(ax, (xs[i] + BW / 2, pm_y), (xs[i] + BW / 2, ROW_Y + BH + 7.5),
              ls="dashed", color=MUTED, lw=1.1)
    label(ax, xs[2] + BW / 2 + 6.5, (pm_y + ROW_Y + BH) / 2 + 2, "invoke",
          size=7.0, color=MUTED)

    # Advisory quality report -- a terminal node, not a feedback edge
    qx = xs[6] + BW / 2
    box(ax, qx - 13, 20, 26, 12, C_TERM, ls="dotted", lw=1.4)
    label(ax, qx, 28.6, "quality report", size=8, weight="bold")
    label(ax, qx, 25.2, "displayed to user;", size=7, style="italic", color="#555555")
    label(ax, qx, 22.6, "no agent consumes it", size=7, style="italic", color="#555555")
    arrow(ax, (qx, ROW_Y), (qx, 32), color=MUTED, lw=1.3)

    # Fallback path -- spans exactly the agents that have one, so every dashed
    # arrow lands inside the box it points at.
    fb_agents = (3, 4, 5, 7)
    fb_x = xs[min(fb_agents)] - 2
    fb_w = (xs[max(fb_agents)] + BW + 2) - fb_x
    fb_y, fb_h = 5, 11
    box(ax, fb_x, fb_y, fb_w, fb_h, C_SIDE, ls="dashed", lw=1.1, ec=MUTED)
    label(ax, fb_x + fb_w / 2, fb_y + fb_h - 4.0,
          "Direct Anthropic Messages API  —  fallback, no role framing",
          size=9, weight="bold", color="#444444")
    label(ax, fb_x + fb_w / 2, fb_y + 3.2,
          "measured activations: 0 across 10 complete runs",
          size=7.8, style="italic", color="#555555")
    for i in fb_agents:
        x_mid = xs[i] + BW / 2
        arrow(ax, (x_mid, ROW_Y), (x_mid, fb_y + fb_h), ls="dashed", color=MUTED, lw=1.0)

    label(ax, 111, 89,
          "Solid edges carry schema-validated JSON in fixed order; the pipeline is strictly linear, with no feedback edges.",
          size=9, style="italic", color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "agent_interaction.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3 -- designed organization (assignment scope, not the paper's scope)
# ---------------------------------------------------------------------------
#
# The project brief specifies an autonomous software development ORGANISATION
# covering marketing, customer care, finance, testing, and deployment as well as
# the software-engineering activities. The prototype implements the engineering
# spine only. These two figures therefore show the full designed organisation
# with the implemented subset visually distinguished -- solid and coloured for
# built, dashed and grey for designed-but-not-implemented. Never present them
# without that distinction: an undifferentiated diagram claims agents that do
# not exist.

C_BUILT = "#d8eef0"
C_BUILT2 = "#fae3cd"
C_PLANNED = "#f4f4f4"


def _unit(ax, x, y, w, h, text, built: bool, *, size=8.0, face=None):
    box(ax, x, y, w, h, face or (C_BUILT if built else C_PLANNED),
        lw=1.4 if built else 1.0,
        ls="solid" if built else "dashed",
        ec=EDGE if built else "#999999")
    label(ax, x + w / 2, y + h / 2, text, size=size,
          weight="bold" if built else "normal",
          color="#111111" if built else "#777777")


def _legend(ax, x, y):
    """Swatch-and-caption legend. Captions are LEFT aligned and offset clear of
    the swatch; centring them (the default in `label`) overlaps the box."""
    box(ax, x, y + 3.2, 3.0, 2.4, C_BUILT, lw=1.4)
    ax.text(x + 4.5, y + 4.4, "implemented and evaluated",
            ha="left", va="center", fontsize=8, color="#111111", zorder=4)
    box(ax, x, y, 3.0, 2.4, C_PLANNED, lw=1.0, ls="dashed", ec="#999999")
    ax.text(x + 4.5, y + 1.2, "designed, not implemented",
            ha="left", va="center", fontsize=8, color="#777777", zorder=4)


def figure_organization(dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 9.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.axis("off")

    label(ax, 50, 101, "Autonomous Software Development Organisation",
          size=13, weight="bold")

    # --- customer interface -------------------------------------------------
    box(ax, 6, 88, 88, 9, C_UI)
    label(ax, 50, 94.6, "Customer Interface Layer", size=10.5, weight="bold")
    _unit(ax, 9, 89, 19, 4.2, "Request intake", True)
    _unit(ax, 30, 89, 19, 4.2, "Requirement\nelicitation", True, size=7.2)
    _unit(ax, 51, 89, 19, 4.2, "Customer portal", False)
    _unit(ax, 72, 89, 19, 4.2, "Order / billing", False)

    # --- orchestration ------------------------------------------------------
    box(ax, 6, 76, 88, 9, C_ORCH)
    label(ax, 50, 82.6, "Orchestration Layer", size=10.5, weight="bold")
    _unit(ax, 12, 77, 24, 4.2, "Project Manager Agent", True)
    _unit(ax, 38, 77, 24, 4.2, "Workflow state\n+ artifact store", True, size=7.2)
    _unit(ax, 64, 77, 24, 4.2, "Resource scheduler", False)

    # --- business agents ----------------------------------------------------
    box(ax, 6, 60, 42, 13, C_GOV)
    label(ax, 27, 70.6, "Business Agents", size=10.5, weight="bold")
    _unit(ax,  9, 65.2, 17.5, 4.2, "Marketing", False)
    _unit(ax, 28, 65.2, 17.5, 4.2, "Finance", False)
    _unit(ax,  9, 60.6, 17.5, 4.2, "Customer care", False)
    _unit(ax, 28, 60.6, 17.5, 4.2, "Legal / compliance", False, size=7.2)

    # --- technical agents ---------------------------------------------------
    box(ax, 52, 60, 42, 13, C_CORE)
    label(ax, 73, 70.6, "Technical Agents", size=10.5, weight="bold")
    _unit(ax, 55, 65.2, 12.0, 4.2, "Requirements", True, size=7.0)
    _unit(ax, 68, 65.2, 12.0, 4.2, "Design", True, size=7.0)
    _unit(ax, 81, 65.2, 11.0, 4.2, "Database", True, size=7.0)
    _unit(ax, 55, 60.6, 12.0, 4.2, "Planning", True, size=7.0)
    _unit(ax, 68, 60.6, 12.0, 4.2, "Quality", True, size=7.0)
    _unit(ax, 81, 60.6, 11.0, 4.2, "Support", True, size=7.0)

    # --- build & release ----------------------------------------------------
    box(ax, 6, 46, 88, 11, C_REND)
    label(ax, 50, 54.6, "Build and Release", size=10.5, weight="bold")
    _unit(ax,  9, 47.4, 26, 5.0,
          "Deterministic renderer\n(scaffold generation)", True, size=7.4)
    _unit(ax, 37, 47.4, 26, 5.0, "Testing agent", False)
    _unit(ax, 65, 47.4, 26, 5.0, "Deployment agent", False)

    # --- knowledge ----------------------------------------------------------
    box(ax, 6, 34, 88, 9, C_SIDE)
    label(ax, 50, 40.6, "Knowledge and Data Layer", size=10.5, weight="bold")
    _unit(ax,  9, 35, 26, 4.2, "Project artifact store", True, size=7.4)
    _unit(ax, 37, 35, 26, 4.2, "Customer database", False)
    _unit(ax, 65, 35, 26, 4.2, "Market intelligence", False)

    # --- infrastructure -----------------------------------------------------
    box(ax, 6, 24, 88, 8, C_APP)
    label(ax, 50, 29.8, "Infrastructure Layer", size=10.5, weight="bold")
    _unit(ax,  9, 24.8, 26, 4.0, "Local / cloud runtime", True, size=7.4)
    _unit(ax, 37, 24.8, 26, 4.0, "Container platform", False)
    _unit(ax, 65, 24.8, 26, 4.0, "Monitoring", False)

    for y1, y2 in [(88, 85), (76, 73), (60, 57), (46, 43), (34, 32)]:
        arrow(ax, (50, y1), (50, y2))

    _legend(ax, 6, 14)
    label(ax, 50, 8,
          "The prototype implements the software-engineering spine; the remaining organisational\n"
          "functions are specified at the design level and are the surface for future work.",
          size=8.5, style="italic", color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "organization_architecture.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_organization_interaction(dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(15.0, 7.0))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 92)
    ax.axis("off")

    label(ax, 100, 88, "Organisation-level agent interaction",
          size=12, weight="bold")

    BH, BW = 11, 21
    ROW = 46

    _unit(ax,   4, ROW, 17, BH, "Customer", True, size=8.5, face=C_SIDE)
    _unit(ax,  27, ROW, BW, BH, "Marketing\nAgent", False, size=8)
    _unit(ax,  54, ROW, BW, BH, "Project Manager\nAgent", True, size=8)
    _unit(ax,  81, ROW, BW, BH, "Technical agent\nspine (6)", True, size=8)
    _unit(ax, 108, ROW, BW, BH, "Testing\nAgent", False, size=8)
    _unit(ax, 135, ROW, BW, BH, "Deployment\nAgent", False, size=8)
    _unit(ax, 162, ROW, BW, BH, "Customer care\nAgent", False, size=8)

    xs = [4, 27, 54, 81, 108, 135, 162]
    ws = [17, BW, BW, BW, BW, BW, BW]
    lbl = ["lead", "brief", "specs", "artifacts", "tested build", "live app"]
    for i in range(len(xs) - 1):
        built = i in (1, 2)
        arrow(ax, (xs[i] + ws[i], ROW + BH / 2), (xs[i + 1], ROW + BH / 2),
              ls="solid" if built else "dashed",
              color=EDGE if built else "#999999",
              lw=1.5 if built else 1.1)
        label(ax, (xs[i] + ws[i] + xs[i + 1]) / 2, ROW + BH + 3.2, lbl[i],
              size=7, color="#444444" if built else "#888888")

    _unit(ax,  54, 22, BW, 9, "Finance\nAgent", False, size=8)
    _unit(ax,  81, 22, BW, 9, "Legal /\nCompliance", False, size=8)
    _unit(ax, 108, 22, BW, 9, "Deterministic\nrenderer", True, size=7.6, face=C_REND)
    for x in (54, 81):
        arrow(ax, (x + BW / 2, ROW), (x + BW / 2, 31), ls="dashed",
              color="#999999", lw=1.1)
    arrow(ax, (81 + BW / 2 + 6, ROW), (108 + BW / 2, 31), color=EDGE, lw=1.4)

    _legend(ax, 4, 6)
    label(ax, 120, 10,
          "Solid: implemented and evaluated in this work.   Dashed: specified, not implemented.",
          size=8.5, style="italic", color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "organization_interaction.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the project's figures.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="300 for submission, 600 for print (default: 300)")
    args = parser.parse_args()
    figures = (
        figure_architecture(args.dpi),          # paper: implemented architecture
        figure_interaction(args.dpi),           # paper: implemented pipeline
        figure_organization(args.dpi),          # assignment: designed organisation
        figure_organization_interaction(args.dpi),
    )
    for path in figures:
        size_kb = path.stat().st_size // 1024
        print(f"wrote {path}  ({size_kb} KB @ {args.dpi} dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
