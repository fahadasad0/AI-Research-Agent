"""Publication-style figures drawn from the *actual* data of a run (no invented numbers)."""
from __future__ import annotations

import io
import textwrap
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless servers such as Streamlit Cloud
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIG_W = 3.45  # inches: one column of the two-column PDF layout
BLUE, GREY, LIGHT = "#1f4e79", "#555555", "#eef2f7"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 6.6, "axes.titlesize": 7,
        "axes.edgecolor": GREY, "axes.linewidth": 0.6, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "figure.dpi": 100,
    })


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def figure_flow(stats: dict, start_label: str) -> bytes:
    """Study identification and screening flow diagram (PRISMA-style) with real counts."""
    _style()
    fig, ax = plt.subplots(figsize=(FIG_W, 3.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 100)
    ax.axis("off")

    src = stats.get("raw_by_source", {})
    src_line = "; ".join(f"{k} {v}" for k, v in src.items()) or "no records"
    main = [
        (80, f"Records identified\nN = {stats['raw_total']}\n({src_line})"),
        (53, f"Unique records, {start_label}\nwith abstract\nn = {stats['unique']}"),
        (26, f"Topically relevant\n(keyword screening)\nn = {stats['screened']}"),
        (0, f"Included in synthesis\nn = {stats['included']}"),
    ]
    side = [
        (76, f"Excluded:\noutside date window\nor no abstract: {stats['dropped_date_or_abstract']}\nduplicates: {stats['duplicates']}"),
        (49, f"Excluded:\nnot relevant to\ntopic keywords: {stats['irrelevant']}"),
        (22.5, f"Excluded:\nbelow ranking\ncut-off: {stats['cutoff']}"),
    ]
    box_w, box_h = 56, 19
    for y, text in main:
        ax.add_patch(FancyBboxPatch((2, y), box_w, box_h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=LIGHT, ec=BLUE, lw=0.8))
        ax.text(2 + box_w / 2, y + box_h / 2, text, ha="center", va="center", fontsize=6)
    for (y, text) in side:
        ax.add_patch(FancyBboxPatch((64, y - 9), 34, 18, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc="white", ec=GREY, lw=0.6))
        ax.text(81, y, text, ha="center", va="center", fontsize=5.5, color=GREY)
        ax.annotate("", xy=(63.4, y), xytext=(30, y), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=GREY))
    for (y, _), (y2, _) in zip(main[:-1], main[1:]):
        ax.annotate("", xy=(30, y2 + box_h + 0.8), xytext=(30, y - 0.2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=BLUE))
    return _png(fig)


def _quarter(d) -> str:
    return f"{d.year} Q{(d.month - 1) // 3 + 1}"


def figure_timeline(papers: list) -> bytes:
    """Included studies per publication quarter."""
    _style()
    counts = Counter(_quarter(p.published) for p in papers)
    first, last = min(p.published for p in papers), max(p.published for p in papers)
    labels, y, q = [], first.year, (first.month - 1) // 3 + 1
    while (y, q) <= (last.year, (last.month - 1) // 3 + 1):
        labels.append(f"{y} Q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    values = [counts.get(l, 0) for l in labels]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.15))
    bars = ax.bar(range(len(labels)), values, color=BLUE, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of studies")
    ax.set_ylim(0, max(values) * 1.25 + 0.5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    for bar, v in zip(bars, values):
        if v:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, str(v), ha="center", va="bottom", fontsize=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    return _png(fig)


def figure_themes(themes: list) -> bytes:
    """Studies per thematic cluster. ``themes`` is a list of (name, ids, summary)."""
    _style()
    rows = sorted(((n, len(ids)) for n, ids, _ in themes), key=lambda r: r[1])
    names = ["\n".join(textwrap.wrap(n, 30)) for n, _ in rows]
    fig, ax = plt.subplots(figsize=(FIG_W, 0.55 + 0.42 * len(rows)))
    bars = ax.barh(range(len(rows)), [c for _, c in rows], color=BLUE, height=0.62)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Number of studies")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(0, max(c for _, c in rows) * 1.2 + 0.5)
    for bar, (_, c) in zip(bars, rows):
        ax.text(c + 0.05, bar.get_y() + bar.get_height() / 2, str(c), va="center", fontsize=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    return _png(fig)
