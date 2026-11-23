"""Build the A1 poster from a completed notebook run."""
import argparse
import io
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from matplotlib import get_data_path
from matplotlib.figure import Figure
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = 841 * mm, 594 * mm
MARGIN = 62
INK = "#172C43"
GRAY = "#4E6272"
TEAL = "#106F76"
GOLD = "#A35616"
LINE = "#C8D8DC"
PALE = "#EDF5F4"
PAPER = "#F7F9F7"
DARK = "#112B3E"
FONT = "PosterSans"
FONT_BOLD = "PosterSans-Bold"


def register_fonts():
    ttf_dir = Path(get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont(FONT, str(ttf_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(ttf_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_BOLD)


def rectangle(canvas, x, top, width, height, fill):
    canvas.setFillColor(HexColor(fill))
    canvas.rect(x, PAGE_H - top - height, width, height, stroke=0, fill=1)


def rule(canvas, x1, x2, top, color=LINE, width=1.5):
    canvas.setStrokeColor(HexColor(color))
    canvas.setLineWidth(width)
    canvas.line(x1, PAGE_H - top, x2, PAGE_H - top)


def text(canvas, value, x, top, size, color=INK, bold=False, align="left"):
    canvas.setFont(FONT_BOLD if bold else FONT, size)
    canvas.setFillColor(HexColor(color))
    if align == "right":
        canvas.drawRightString(x, PAGE_H - top - size * 0.83, value)
    else:
        canvas.drawString(x, PAGE_H - top - size * 0.83, value)


def paragraph(canvas, value, x, top, width, size=25, color=GRAY, max_height=None):
    style = ParagraphStyle("poster", fontName=FONT, fontSize=size, leading=size * 1.24,
                           textColor=HexColor(color))
    p = Paragraph(value, style)
    _, height = p.wrap(width, PAGE_H)
    if max_height is not None and height > max_height:
        raise ValueError(f"Text exceeds its space by {height - max_height:.1f} pt: {value}")
    p.drawOn(canvas, x, PAGE_H - top - height)
    return height


def heading(canvas, title, x, top, width):
    text(canvas, title, x, top, 39, bold=True)
    rule(canvas, x, x + width, top + 49)


def draw_table(canvas, x, top, widths, headers, rows, row_height, size=23, right_columns=()):
    width = sum(widths)
    rectangle(canvas, x, top, width, row_height, DARK)
    for j, value in enumerate(headers):
        left = x + sum(widths[:j])
        right = j in right_columns
        text(canvas, value, left + widths[j] - 14 if right else left + 14, top + 9,
             size, "#FFFFFF", bold=True, align="right" if right else "left")
    for i, row in enumerate(rows, 1):
        y = top + i * row_height
        rectangle(canvas, x, y, width, row_height, PALE if i % 2 else "#FFFFFF")
        for j, value in enumerate(row):
            left = x + sum(widths[:j])
            if j in right_columns:
                text(canvas, str(value), left + widths[j] - 14, y + 9, size, align="right")
            else:
                paragraph(canvas, escape(str(value)), left + 14, y + 4,
                          widths[j] - 28, size=size, color=INK, max_height=row_height - 4)


def arrow(canvas, x1, x2, top):
    canvas.setStrokeColor(HexColor(TEAL))
    canvas.setLineWidth(2.5)
    y = PAGE_H - top
    canvas.line(x1, y, x2, y)
    canvas.line(x2 - 9, y + 6, x2, y)
    canvas.line(x2 - 9, y - 6, x2, y)


def pipeline_lane(canvas, label_lines, boxes, top):
    left = MARGIN + 214
    gap = 35
    box_width = (PAGE_W - MARGIN - left - gap * 3) / 4
    for index, label in enumerate(label_lines):
        text(canvas, label, MARGIN, top + 25 + index * 27, 23, color=TEAL, bold=True)
    for index, (title, description) in enumerate(boxes):
        x = left + index * (box_width + gap)
        rectangle(canvas, x, top, box_width, 94, PALE)
        rectangle(canvas, x, top, 5, 94, TEAL)
        text(canvas, title, x + 17, top + 14, 27, bold=True)
        paragraph(canvas, description, x + 17, top + 49, box_width - 33, size=21, max_height=46)
        if index < 3:
            arrow(canvas, x + box_width + 4, x + box_width + gap - 5, top + 47)


def speedup_chart(bench, serial_fraction):
    figure = Figure(figsize=(11.7, 5.65), dpi=220, facecolor=PAPER)
    ax = figure.add_subplot(facecolor=PAPER)
    workers = bench["workers"]
    ax.plot(workers, workers, "--", lw=2, color="#7D8794", label="Ideal")
    if 0 < serial_fraction < 1:
        grid = np.linspace(workers.min(), workers.max(), 100)
        model = 1 / (serial_fraction + (1 - serial_fraction) / grid)
        ax.plot(grid, model, ":", lw=2, color="#75558C", label="Amdahl fit")
    ax.errorbar(
        workers, bench["speedup"], yerr=bench["speedup"] * bench["std_s"] / bench["mean_s"],
        marker="o", color=TEAL, linewidth=3, capsize=4, markersize=8, label="Warm pool",
    )
    ax.plot(workers, bench["speedup_incl_startup"], "s-", color=GOLD, lw=3,
            markersize=8, label="Fresh pool")
    ax.axhline(1, linewidth=1, color="#85939D")
    ax.set_xticks(workers)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Worker processes", color=INK, fontsize=17, labelpad=10)
    ax.set_ylabel("Speedup vs sequential", color=INK, fontsize=17, labelpad=13)
    ax.tick_params(axis="both", labelsize=16, colors=INK)
    ax.grid(axis="y", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A3AC")
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=15, labelcolor=INK)
    figure.subplots_adjust(left=0.105, right=0.982, bottom=0.175, top=0.98)
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=220, facecolor=PAPER)
    stream.seek(0)
    return ImageReader(stream)


def build_poster(results, output):
    register_fonts()

    def read(name, **kwargs):
        return pd.read_csv(results / name, **kwargs)

    summary = read("run_summary.csv", dtype=str, keep_default_na=False).set_index("setting")["value"]
    env = read("environment.csv", dtype=str, keep_default_na=False).set_index("setting")["value"]
    checks = ("aggregate verification", "feature equivalence", "search equivalence", "seeded repeatability")
    if not all(summary[check] == "Passed" for check in checks):
        raise ValueError("The selected run did not pass all notebook verification checks.")

    cleaning = read("cleaning_steps.csv")
    thresholds = read("cleaning_thresholds.csv").set_index("statistic")["value"]
    splits = read("time_split.csv").set_index("split")
    tuning = read("tuning_split.csv").set_index("split")["tuning_rows"]
    model = read("model_results.csv").set_index("metric")
    full_bench = read("benchmark_data_parallel.csv")
    sequential = full_bench.loc[full_bench.version == "sequential loop"].iloc[0]
    bench = full_bench.loc[full_bench.version == "process pool"].sort_values("workers")
    last = bench.iloc[-1]
    serial_frac = bench["karp_flatt_serial_frac"].mean()
    grid = read("benchmark_hyperparameter_search.csv").sort_values("workers")
    feature = read("benchmark_feature_extraction.csv").set_index("version")
    differences = read("nondeterminism_summary.csv")
    groups = len(read("feature_group_times.csv"))
    trials = len(read("grid_search_results.csv"))

    output.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(output), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Parallel Taxi Tip Prediction | CS4207 Group 8")
    c.setAuthor("CS4207 Group 8 | University of Limerick")
    c.setSubject("Data and task parallelism on January 2026 NYC Yellow Taxi trips")
    rectangle(c, 0, 0, PAGE_W, PAGE_H, PAPER)
    rectangle(c, 0, 0, PAGE_W, 241, DARK)
    rectangle(c, 0, 241, PAGE_W, 8, "#E3B44F")
    text(c, "UNIVERSITY OF LIMERICK  /  CS4207 CODING CHALLENGE", MARGIN, 41, 24, "#9AD9D5", bold=True)
    text(c, "GROUP 8", PAGE_W - MARGIN, 41, 24, "#F0C96E", bold=True, align="right")
    text(c, "Parallel taxi tip prediction", MARGIN, 85, 83, "#FFFFFF", bold=True)
    text(c, "Data and task parallelism on January 2026 NYC Yellow Taxi trips",
         MARGIN + 3, 184, 32, "#C7E2E5")

    heading(c, "The experiment", MARGIN, 279, PAGE_W - 2 * MARGIN)
    pipeline_lane(c, ("DATA", "PARALLELISM"), [
        (f"{summary['shards']} pickup-day shards", "One task per day"),
        ("Read, clean and aggregate", "Apply the same rules in each worker"),
        ("Merge partial results", "Sum by pickup zone and hour"),
        ("Verify the aggregates", "Match sequential and direct totals"),
    ], 348)
    pipeline_lane(c, ("TASK", "PARALLELISM"), [
        ("Clean card trips", "Train, validate and test by date"),
        (f"Build {groups} feature groups", "Join in a fixed order and check equality"),
        (f"Fit {trials} independent trials", "Select by validation ROC-AUC"),
        ("Evaluate on later trips", "Logistic regression vs always tip"),
    ], 457)
    paragraph(c, "<b>Will a passenger leave a recorded tip on a completed card trip?</b> "
              "Trip features are available before tip entry. The benchmark uses CPU workers.",
              MARGIN, 563, PAGE_W - 2 * MARGIN, size=25, max_height=38)

    left_w, mid_w = 570, 849
    gutter = 40
    left = MARGIN
    middle = left + left_w + gutter
    right = middle + mid_w + gutter
    right_w = PAGE_W - MARGIN - right
    main_y = 627

    heading(c, "Dataset and cleaning", left, main_y, left_w)
    paragraph(c, "Cash tips are unrecorded. Keeping card payments also excludes Flex Fare rides with missing fields.",
              left, 690, left_w, size=25, max_height=94)
    clean_labels = [
        "Raw trips", "Card payments", "January pickups", "Distance > 0 mi",
        "Distance ≤ 100 mi", "Meter fare $0–$500*", "Duration 1–180 min",
        "Average speed ≤ 80 mph", "Recorded tip ≥ $0",
    ]
    if len(cleaning) != len(clean_labels):
        raise ValueError("Poster cleaning labels no longer match the notebook steps.")
    clean_rows = [(label, f"{count:,}") for label, count in zip(clean_labels, cleaning.rows_after)]
    draw_table(c, left, 795, [left_w - 160, 160], ["After each step", "Trips"], clean_rows,
               row_height=32, size=20, right_columns=(1,))
    raw_max = thresholds["Maximum distance in raw file (mi)"]
    card_max = thresholds["Maximum distance in January card trips (mi)"]
    paragraph(
        c, f"*Meter fares must be strictly positive. Maximum raw distance was <b>{raw_max:,.2f} mi</b>. "
        f"Among January card trips it was <b>{card_max:,.2f} mi</b>.",
        left, 1126, left_w, size=21, max_height=82,
    )
    paragraph(c, "<b>January split</b>  ·  1–17 train  /  18–24 validation  /  25–31 test",
              left, 1211, left_w, size=22, color=INK, max_height=61)

    heading(c, f"Speedup on {summary['shards']} day shards", middle, main_y, mid_w)
    paragraph(c, f"Identical work per run. One warm-up and <b>{summary['benchmark repeats']} timed repeats</b>.",
              middle, 690, mid_w, size=25, max_height=66)
    c.drawImage(speedup_chart(bench, serial_frac), middle, PAGE_H - 738 - 407, mid_w, 407,
                preserveAspectRatio=True, anchor="c", mask="auto")
    count = int(last.workers)
    paragraph(
        c, f"<b>At {count} workers</b>, the warm pool reached {last.speedup:.2f}× ({last.efficiency:.0%} efficiency) "
        f"and {last.speedup_incl_startup:.2f}× with a fresh pool. "
        f"The sequential loop took {sequential.mean_s:.3f} s.",
        middle, 1152, mid_w, size=25, color=INK, max_height=70,
    )
    estimate = f"The effective parallel fraction was {(1 - serial_frac):.1%}. " if 0 < serial_frac < 1 else ""
    paragraph(c, estimate + "Worker startup, shared file reads and merging partial tables affect scaling.",
              middle, 1221, mid_w, size=21, max_height=58)

    heading(c, "Task parallelism and model", right, main_y, right_w)
    paragraph(
        c, f"<b>Model search</b><br/>{grid.iloc[0].seconds:.2f} s sequential vs "
        f"{grid.iloc[-1].seconds:.2f} s with {int(grid.iloc[-1].workers)} workers "
        f"({grid.iloc[-1].speedup:.2f}×). Each trial uses the same {tuning.at['train']:,} training "
        f"and {tuning.at['validation']:,} validation rides. Unequal trial times limit speedup.",
        right, 688, right_w, size=25, max_height=155,
    )
    paragraph(
        c, f"<b>Feature extraction</b><br/>{feature.at['sequential', 'mean_s']:.3f} s sequential, "
        f"{feature.at['fresh pool', 'mean_s']:.3f} s with a fresh pool and "
        f"{feature.at['warm pool', 'mean_s']:.3f} s with a warm pool. All resulting feature tables match.",
        right, 854, right_w, size=25, max_height=160,
    )
    prevalence = splits.at["test", "tip_rate"]
    paragraph(c, f"<b>Held-out test week</b>  ·  {int(splits.at['test', 'rows']):,} card trips; "
              f"{prevalence:.1%} had a tip.", right, 1020, right_w, size=23, max_height=60)
    model_rows = [
        (label, f"{model.at[key, 'selected logistic (test)']:.3f}",
         f"{model.at[key, 'majority baseline (test)']:.3f}")
        for key, label in [("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced accuracy"),
                           ("f1", "F1 (tipped class)"), ("accuracy", "Accuracy")]
    ]
    draw_table(c, right, 1081, [right_w - 372, 170, 202], ["Metric", "Logistic", "Always tip"],
               model_rows, row_height=34, size=22, right_columns=(1, 2))
    paragraph(c, "Model settings were selected on the previous week’s validation data.",
              right, 1257, right_w, size=21, max_height=54)

    heading(c, "Why repeated runs can differ", MARGIN, 1321, PAGE_W - 2 * MARGIN)
    inner_w = PAGE_W - 2 * MARGIN
    reproducibility_rows = list(differences[["source", "before", "fix", "after"]].itertuples(index=False, name=None))
    draw_table(c, MARGIN, 1386, [410, 605, 625, inner_w - 1640], ["Source", "Before", "Fix", "After"],
               reproducibility_rows, row_height=64, size=24)

    rule(c, MARGIN, PAGE_W - MARGIN, 1598, color=TEAL, width=2)
    cpu = env["cpu_model"].replace(" 96-Core Processor", "")
    host = "Local Linux run" if "Linux" in env["platform"] else "Local run"
    if summary["slurm job"] != "-":
        host = f"Slurm job {summary['slurm job']}"
    text(c, f"{host}  ·  {cpu}  ·  {summary['usable_cpus']} logical CPUs available  ·  CPU only  ·  "
         f"Notebook runtime {float(summary['total notebook run time (s)']):.1f} s",
         MARGIN, 1611, 20, color=INK, bold=True)
    text(c, "NYC TLC Yellow Taxi Trip Records, January 2026   |   Notebook sections 1–7 and exported tables",
         MARGIN, 1640, 18, color=GRAY)
    c.linkURL("https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
              (MARGIN, PAGE_H - 1661, MARGIN + 495, PAGE_H - 1638), relative=0)
    c.showPage()
    c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "poster.pdf")
    args = parser.parse_args()
    build_poster(args.results, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
