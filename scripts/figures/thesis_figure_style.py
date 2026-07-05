"""Shared matplotlib style so every generated figure renders text at the thesis
font size (IEEEtran, 10pt, two-column).

THE RULE: a figure is scaled by LaTeX to the \\includegraphics width. If you draw
at 7 in and include at \\linewidth (~3.49 in in a 10pt IEEE column), every label
shrinks by 0.5x. To get TRUE 10pt text, draw the figure at its FINAL display
width and set font.size=10 — then \\includegraphics[width=\\columnwidth] (single
column) or width=\\textwidth (figure*, full width) scales by 1.0 and the text is
exactly 10pt.

IEEEtran 10pt geometry:
  \\columnwidth = 252.0 pt = 3.487 in   (single-column figure)
  \\textwidth   = 516.0 pt = 7.139 in   (figure* spanning both columns)

Usage:
    from thesis_figure_style import use_thesis_style, col_fig, text_fig
    use_thesis_style()                 # call once, before plotting
    fig, ax = text_fig(height=3.0)     # full-width figure* -> width=\\textwidth
    fig, ax = col_fig(height=2.4)      # single-column     -> width=\\columnwidth
    ...
    fig.savefig("figures/fig_x.png")   # dpi/layout already set
And in LaTeX include with the MATCHING width (no up/down-scaling):
    \\includegraphics[width=\\columnwidth]{...}   % for col_fig
    \\includegraphics[width=\\textwidth]{...}      % for text_fig (inside figure*)
"""
from __future__ import annotations

COL_W = 3.487     # \columnwidth in inches (IEEEtran, 10pt)
TEXT_W = 7.139    # \textwidth in inches (figure*)
BASE_PT = 10      # thesis body font size


def use_thesis_style(font_pt: int = BASE_PT) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",          # IEEEtran is Times-like; serif matches the body text
        "mathtext.fontset": "stix",
        "font.size": font_pt,            # body text size -> TRUE document size at scale 1.0
        "axes.titlesize": font_pt,
        "axes.labelsize": font_pt,
        "xtick.labelsize": font_pt - 1,  # ticks one step down, still readable, IEEE-conventional
        "ytick.labelsize": font_pt - 1,
        "legend.fontsize": font_pt - 1,
        "figure.titlesize": font_pt,
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,              # crisp in print; PNG file size is fine at these sizes
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
    })


def col_fig(height: float = 2.4, **kw):
    """Single-column figure -> include with width=\\columnwidth (scale 1.0)."""
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(COL_W, height), **kw)


def text_fig(height: float = 3.0, **kw):
    """Full-width figure* -> include with width=\\textwidth (scale 1.0)."""
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(TEXT_W, height), **kw)
