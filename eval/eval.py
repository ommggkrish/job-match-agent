"""Measure how often the agent gets the labels in expected.csv right.

Run (needs GEMINI_API_KEY in .env):
    python eval/eval.py              # runs every ad that has no result yet for this model
    python eval/eval.py --only 02 03 # (re-)run single ads
    python eval/eval.py --fresh      # forget saved results for this model and start again

The eval can be spread over several days: each finished ad is saved to eval/results.csv
right away, and the next run continues with the missing ads. This matters on the Gemini
free tier, which allows only about 20 requests per model per day (one ad needs ~5).

API failures (quota, overload, network) skip the ad instead of counting it as wrong.
A quota error (429) stops the run. A real agent failure (e.g. invalid JSON) counts as wrong.

Labels compared per ad: fit (strong/possible fit = yes), location_ok, hours_ok, language_ok.
The summary over all saved results for this model is written to eval/results.md.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from google.genai.errors import APIError  # noqa: E402

try:  # network errors from the HTTP library used by google-genai
    from httpx2 import TransportError as NetworkError  # noqa: E402
except ImportError:
    from httpx import TransportError as NetworkError  # noqa: E402

from agent.agent import run_agent  # noqa: E402
from agent.llm import GeminiClient  # noqa: E402

EVAL_DIR = ROOT / "eval"
EXPECTED_CSV = EVAL_DIR / "expected.csv"
RESULTS_CSV = EVAL_DIR / "results.csv"
RESULTS_MD = EVAL_DIR / "results.md"
LABELS = ["fit", "location_ok", "hours_ok", "language_ok"]
RESULT_FIELDS = ["file", "model", "date", "score", "verdict", *LABELS]


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def predict(report) -> dict[str, str]:
    c = report.constraints
    return {
        "fit": yes_no(report.verdict in ("strong fit", "possible fit")),
        "location_ok": yes_no(c.location_ok) if c else "?",
        "hours_ok": yes_no(c.hours_ok) if c else "?",
        "language_ok": yes_no(c.language_ok) if c else "?",
    }


# ------------------------------------------------------------ saved results

def load_results(path: Path = RESULTS_CSV) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def save_results(results: list[dict], path: Path = RESULTS_CSV) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["model"], r["file"])))


def summarize(expected: list[dict], results: list[dict], model: str) -> list[str]:
    """Markdown summary + table for one model. Pure function, so it is unit-tested."""
    by_file = {r["file"]: r for r in results if r["model"] == model}
    done = [row for row in expected if row["file"] in by_file]
    missing = [row["file"] for row in expected if row["file"] not in by_file]
    if not done:
        return [f"# Eval – model `{model}`", "", "No results yet."]

    correct = {label: 0 for label in LABELS}
    table = ["| Ad | Score | Verdict | " + " | ".join(LABELS) + " |", "|---|---|---|" + "---|" * len(LABELS)]
    for row in done:
        res = by_file[row["file"]]
        cells = []
        for label in LABELS:
            ok = res[label] == row[label]
            correct[label] += ok
            cells.append(res[label] if ok else f"**{res[label]}** (expected {row[label]})")
        table.append(f"| {row['file']} | {res['score']} | {res['verdict']} | " + " | ".join(cells) + " |")

    n = len(done)
    total = sum(correct.values())
    dates = sorted({by_file[row["file"]]["date"] for row in done})
    lines = [
        f"# Eval – model `{model}`",
        "",
        f"- ads evaluated: {n} of {len(expected)}" + (f" (still missing: {', '.join(missing)})" if missing else ""),
        f"- run on: {', '.join(dates)}",
    ]
    lines += [f"- {label}: {correct[label]}/{n} ({correct[label] / n:.0%})" for label in LABELS]
    lines += [f"- **overall: {total}/{n * len(LABELS)} ({total / (n * len(LABELS)):.0%})**", "", *table, ""]
    return lines


# ---------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="(re-)run ads whose file name starts with these, e.g. --only 02 03")
    parser.add_argument("--fresh", action="store_true", help="delete saved results for this model first")
    parser.add_argument("--delay", type=float, default=2, help="seconds to wait between ads")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    llm = GeminiClient()
    expected = list(csv.DictReader(EXPECTED_CSV.open(encoding="utf-8")))
    results = load_results()
    if args.fresh:
        results = [r for r in results if r["model"] != llm.model]

    done = {r["file"] for r in results if r["model"] == llm.model}
    if args.only:
        todo = [row for row in expected if any(row["file"].startswith(p) for p in args.only)]
    else:
        todo = [row for row in expected if row["file"] not in done]
    print(f"Model {llm.model}: {len(done)} ads already done, {len(todo)} to run.\n")

    for i, row in enumerate(todo):
        if i and args.delay:
            time.sleep(args.delay)
        text = (EVAL_DIR / "job_ads" / row["file"]).read_text(encoding="utf-8")
        try:
            report = run_agent(text, llm).report
            pred, score, verdict = predict(report), str(report.score), report.verdict
        except (APIError, NetworkError) as exc:
            # The API failed – that says nothing about the agent, so the ad is skipped.
            print(f"SKIPPED {row['file']}: {str(exc)[:150]}")
            if getattr(exc, "code", None) == 429:
                print("\nDaily quota used up – stopping. Run the same command again tomorrow.")
                break
            continue
        except Exception as exc:  # the agent itself failed – counts as wrong
            print(f"AGENT ERROR on {row['file']}: {exc}")
            pred, score, verdict = {label: "error" for label in LABELS}, "-", "error"

        results = [r for r in results if not (r["model"] == llm.model and r["file"] == row["file"])]
        results.append({"file": row["file"], "model": llm.model, "date": time.strftime("%Y-%m-%d"),
                        "score": score, "verdict": verdict, **pred})
        save_results(results)  # save after every ad, so nothing is lost if a later ad fails
        print(f"{row['file']:45} score={score:>3}  {verdict}")

    summary = summarize(expected, results, llm.model)
    RESULTS_MD.write_text("\n".join(summary), encoding="utf-8")
    print("\n" + "\n".join(summary[:9]))
    print("\nSaved to eval/results.csv and eval/results.md")


if __name__ == "__main__":
    main()
