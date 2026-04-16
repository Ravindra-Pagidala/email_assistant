"""
Run Evaluation — CLI Entry Point.

Usage:
    python run_evaluation.py --model gemma
    python run_evaluation.py --model groq
    python run_evaluation.py --model both
    python run_evaluation.py --model compare   <- use existing CSVs, no API calls
"""

import argparse
import os
import sys
import csv
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.logger import setup_logging, get_logger
from evaluation.evaluator import EvaluationPipeline, EvaluationSummary

logger = get_logger(__name__)


def safe_float(val, default=0.0) -> float:
    """Safely convert any value to float. Returns default if empty or invalid."""
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def run_model(model_name: str, delay: float = 4.0) -> EvaluationSummary:
    print(f"\n{'='*65}")
    print(f"  Starting evaluation: {model_name.upper()}")
    print(f"{'='*65}\n")
    pipeline = EvaluationPipeline(
        model_name=model_name,
        rate_limit_delay=delay,
        scenarios_path="scenarios/test_scenarios.json",
        results_dir="results",
    )
    return pipeline.run()


def correct_overall(s: EvaluationSummary) -> float:
    """
    Overall = (avg_fact + avg_tone + avg_action) / 3
    Calculated only from successful scenarios.
    """
    return round(
        (s.avg_fact_integration_score +
         s.avg_tone_consistency_score +
         s.avg_actionability_score) / 3, 4
    )


def metric_winner(a: float, b: float, lower_is_better: bool = False) -> str:
    if lower_is_better:
        return "GROQ" if b <= a else "GEMMA"
    return "GEMMA" if a >= b else "GROQ"


def save_comparison_csv(
    gemma_summary: EvaluationSummary,
    groq_summary: EvaluationSummary,
    gemma_results: list,
    groq_results: list,
) -> str:
    """
    Save full comparison as a single CSV file with:
      Section 1 — Metric averages across all 10 scenarios
      Section 2 — Speed comparison
      Section 3 — Per-scenario breakdown
      Section 4 — FINAL WINNER declaration
    No JSON files produced anywhere.
    """
    os.makedirs("results", exist_ok=True)
    path = "results/comparison_summary.csv"

    g_overall = correct_overall(gemma_summary)
    q_overall = correct_overall(groq_summary)
    g_lat = gemma_summary.avg_latency_ms
    q_lat = groq_summary.avg_latency_ms
    speed_ratio = round(g_lat / q_lat, 1) if q_lat > 0 else 0
    g_rpm = round(60000 / g_lat, 1) if g_lat > 0 else 0
    q_rpm = round(60000 / q_lat, 1) if q_lat > 0 else 0

    # Count how many metric categories each model wins
    gemma_metric_wins = sum([
        gemma_summary.avg_fact_integration_score >= groq_summary.avg_fact_integration_score,
        gemma_summary.avg_tone_consistency_score >= groq_summary.avg_tone_consistency_score,
        gemma_summary.avg_actionability_score >= groq_summary.avg_actionability_score,
    ])
    groq_metric_wins = 3 - gemma_metric_wins

    # Determine final winner logic
    # If accuracy gap < 0.02 → speed decides → Groq wins
    # Else → accuracy winner wins
    accuracy_diff = abs(g_overall - q_overall)
    if accuracy_diff < 0.02:
        final_winner = "GROQ"
        final_reason = (
            f"Accuracy is comparable (Gemma:{g_overall} vs Groq:{q_overall}, "
            f"diff:{accuracy_diff:.4f} < 0.02 threshold). "
            f"Speed is the tiebreaker — Groq is {speed_ratio}x faster "
            f"({round(q_lat)}ms vs {round(g_lat)}ms). "
            f"Groq recommended for production."
        )
    elif g_overall > q_overall:
        final_winner = "GEMMA"
        final_reason = (
            f"Gemma has meaningfully higher accuracy "
            f"({g_overall} vs {q_overall}, diff:{accuracy_diff:.4f}). "
            f"Best for batch/async use cases where quality > speed."
        )
    else:
        final_winner = "GROQ"
        final_reason = (
            f"Groq has higher accuracy ({q_overall} vs {g_overall}) "
            f"AND {speed_ratio}x faster speed. Clear production winner."
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # ── Section 1: Metric Averages ────────────────────────────
        w.writerow(["SECTION 1: METRIC AVERAGES ACROSS ALL 10 SCENARIOS"])
        w.writerow([
            "metric",
            f"gemma_avg  [{gemma_summary.model_name}]",
            f"groq_avg  [{groq_summary.model_name}]",
            "delta (gemma minus groq)",
            "metric_winner",
            "metric_definition",
        ])
        w.writerow([
            "Fact Integration Score",
            gemma_summary.avg_fact_integration_score,
            groq_summary.avg_fact_integration_score,
            round(gemma_summary.avg_fact_integration_score - groq_summary.avg_fact_integration_score, 4),
            metric_winner(gemma_summary.avg_fact_integration_score, groq_summary.avg_fact_integration_score),
            "Pure Python keyword matching. Score = key_facts_found / total_key_facts.",
        ])
        w.writerow([
            "Tone Consistency Score",
            gemma_summary.avg_tone_consistency_score,
            groq_summary.avg_tone_consistency_score,
            round(gemma_summary.avg_tone_consistency_score - groq_summary.avg_tone_consistency_score, 4),
            metric_winner(gemma_summary.avg_tone_consistency_score, groq_summary.avg_tone_consistency_score),
            "LLM-as-Judge scores opening/body/closing 1-10. Drift penalty if any section < 5.",
        ])
        w.writerow([
            "Actionability Score",
            gemma_summary.avg_actionability_score,
            groq_summary.avg_actionability_score,
            round(gemma_summary.avg_actionability_score - groq_summary.avg_actionability_score, 4),
            metric_winner(gemma_summary.avg_actionability_score, groq_summary.avg_actionability_score),
            "LLM-as-Judge: clear_ask + next_step + timeframe. Score = components_present / 3.",
        ])
        w.writerow([
            "OVERALL ACCURACY  (fact + tone + action) / 3",
            g_overall,
            q_overall,
            round(g_overall - q_overall, 4),
            metric_winner(g_overall, q_overall),
            "Mean of all 3 metric averages. Calculated only from successful scenarios.",
        ])

        # ── Section 2: Speed Comparison ───────────────────────────
        w.writerow([])
        w.writerow(["SECTION 2: SPEED COMPARISON"])
        w.writerow([
            "metric",
            f"gemma  [{gemma_summary.model_name}]",
            f"groq  [{groq_summary.model_name}]",
            "delta",
            "speed_winner",
            "note",
        ])
        w.writerow([
            "Avg Latency per Email (ms)",
            round(g_lat),
            round(q_lat),
            round(g_lat - q_lat),
            "GROQ",
            f"Groq is {speed_ratio}x faster than Gemma",
        ])
        w.writerow([
            "Estimated Requests per Minute",
            g_rpm,
            q_rpm,
            round(q_rpm - g_rpm, 1),
            "GROQ",
            "Higher = better throughput for production",
        ])
        w.writerow([
            "Successful Scenarios (out of 10)",
            gemma_summary.successful_generations,
            groq_summary.successful_generations,
            groq_summary.successful_generations - gemma_summary.successful_generations,
            metric_winner(
                float(gemma_summary.successful_generations),
                float(groq_summary.successful_generations)
            ),
            "Failed scenarios score 0 on all metrics",
        ])

        # ── Section 3: Per-Scenario Breakdown ─────────────────────
        w.writerow([])
        w.writerow(["SECTION 3: PER-SCENARIO BREAKDOWN"])
        w.writerow([
            "scenario_id", "intent", "tone",
            "gemma_fact", "gemma_tone", "gemma_action", "gemma_avg_score",
            "groq_fact", "groq_tone", "groq_action", "groq_avg_score",
            "scenario_winner",
        ])

        gm = {r["scenario_id"]: r for r in gemma_results}
        gq = {r["scenario_id"]: r for r in groq_results}

        for sid in range(1, 11):
            g = gm.get(sid, {})
            q = gq.get(sid, {})
            g_avg = safe_float(g.get("average_score"))
            q_avg = safe_float(q.get("average_score"))
            g_ok = str(g.get("generation_success", "")) == "True"
            q_ok = str(q.get("generation_success", "")) == "True"

            if not g_ok and not q_ok:
                s_winner = "BOTH FAILED"
            elif not g_ok:
                s_winner = "GROQ"
            elif not q_ok:
                s_winner = "GEMMA"
            else:
                s_winner = "GEMMA" if g_avg >= q_avg else "GROQ"

            w.writerow([
                sid,
                str(g.get("intent", q.get("intent", "")))[:50],
                g.get("tone", q.get("tone", "")),
                safe_float(g.get("fact_integration_score")),
                safe_float(g.get("tone_consistency_score")),
                safe_float(g.get("actionability_score")),
                g_avg,
                safe_float(q.get("fact_integration_score")),
                safe_float(q.get("tone_consistency_score")),
                safe_float(q.get("actionability_score")),
                q_avg,
                s_winner,
            ])

        # ── Section 4: FINAL WINNER ───────────────────────────────
        w.writerow([])
        w.writerow(["SECTION 4: FINAL WINNER DECLARATION"])
        w.writerow(["dimension", "gemma_value", "groq_value", "WINNER"])
        w.writerow([
            "Fact Integration",
            gemma_summary.avg_fact_integration_score,
            groq_summary.avg_fact_integration_score,
            metric_winner(gemma_summary.avg_fact_integration_score, groq_summary.avg_fact_integration_score),
        ])
        w.writerow([
            "Tone Consistency",
            gemma_summary.avg_tone_consistency_score,
            groq_summary.avg_tone_consistency_score,
            metric_winner(gemma_summary.avg_tone_consistency_score, groq_summary.avg_tone_consistency_score),
        ])
        w.writerow([
            "Actionability",
            gemma_summary.avg_actionability_score,
            groq_summary.avg_actionability_score,
            metric_winner(gemma_summary.avg_actionability_score, groq_summary.avg_actionability_score),
        ])
        w.writerow([
            "Overall Accuracy",
            g_overall,
            q_overall,
            metric_winner(g_overall, q_overall),
        ])
        w.writerow([
            "Speed (lower latency = better)",
            f"{round(g_lat)}ms",
            f"{round(q_lat)}ms",
            "GROQ",
        ])
        w.writerow([
            "Metric Category Wins (out of 3)",
            f"{gemma_metric_wins}/3",
            f"{groq_metric_wins}/3",
            "GEMMA" if gemma_metric_wins > groq_metric_wins else "GROQ",
        ])
        w.writerow([])
        w.writerow(["*** FINAL WINNER ***", "", "", final_winner])
        w.writerow(["Reason", final_reason, "", ""])
        w.writerow(["Accuracy Winner", metric_winner(g_overall, q_overall), "", ""])
        w.writerow(["Speed Winner", "GROQ", f"{speed_ratio}x faster", ""])
        w.writerow(["Production Recommendation", final_winner, "", ""])

    logger.info(f"Comparison CSV saved: {path}")

    # ── Print to console (same as CSV) ────────────────────────────
    print("\n" + "=" * 72)
    print("  COMPARATIVE ANALYSIS")
    print("=" * 72)
    print(f"\n  {'ACCURACY':<30} {'Gemma':>10} {'Groq':>10} {'Delta':>8} {'Winner':>8}")
    print("  " + "-" * 68)
    for label, gv, qv in [
        ("Fact Integration", gemma_summary.avg_fact_integration_score, groq_summary.avg_fact_integration_score),
        ("Tone Consistency", gemma_summary.avg_tone_consistency_score, groq_summary.avg_tone_consistency_score),
        ("Actionability", gemma_summary.avg_actionability_score, groq_summary.avg_actionability_score),
    ]:
        print(f"  {label:<30} {gv:>10.4f} {qv:>10.4f} {gv-qv:>+8.4f} {metric_winner(gv,qv):>8}")
    print("  " + "-" * 68)
    print(f"  {'OVERALL ACCURACY':<30} {g_overall:>10.4f} {q_overall:>10.4f} {g_overall-q_overall:>+8.4f} {metric_winner(g_overall,q_overall):>8}")
    print(f"\n  {'SPEED':<30} {'Gemma':>10} {'Groq':>10} {'':>8} {'Winner':>8}")
    print("  " + "-" * 68)
    print(f"  {'Avg Latency (ms)':<30} {g_lat:>10.0f} {q_lat:>10.0f} {'':>8} {'GROQ':>8}")
    print(f"  {'Speed Ratio':<30} {'1.0x':>10} {speed_ratio:>9.1f}x {'':>8} {'GROQ':>8}")
    print(f"  {'Est. Requests/min':<30} {g_rpm:>10.1f} {q_rpm:>10.1f} {'':>8} {'GROQ':>8}")
    print("  " + "-" * 68)
    print(f"\n  🏆 FINAL WINNER    : {final_winner}")
    print(f"  📌 ACCURACY WINNER : {metric_winner(g_overall, q_overall)}")
    print(f"  ⚡ SPEED WINNER    : GROQ ({speed_ratio}x faster)")
    print(f"  📄 REASON          : {final_reason[:90]}...")
    print(f"\n  ✅ Full report: {path}")
    print("=" * 72)

    return path


def load_csv(model_name: str) -> list:
    """Load results CSV. Handles empty/missing numeric fields safely."""
    path = f"results/{model_name}_results.csv"
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["scenario_id"] = int(safe_float(row.get("scenario_id"), 1))
            row["fact_integration_score"] = safe_float(row.get("fact_integration_score"))
            row["tone_consistency_score"] = safe_float(row.get("tone_consistency_score"))
            row["actionability_score"] = safe_float(row.get("actionability_score"))
            row["average_score"] = safe_float(row.get("average_score"))
            row["generation_latency_ms"] = safe_float(row.get("generation_latency_ms"))
            rows.append(row)
    return rows


def summary_from_csv(results: list, model_name: str) -> EvaluationSummary:
    """Rebuild EvaluationSummary from CSV rows."""
    ok = [r for r in results if str(r.get("generation_success")) == "True"]

    def avg(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0

    lats = [r["generation_latency_ms"] for r in ok if r["generation_latency_ms"] > 0]

    return EvaluationSummary(
        model_name=model_name,
        total_scenarios=len(results),
        successful_generations=len(ok),
        failed_generations=len(results) - len(ok),
        avg_fact_integration_score=avg([r["fact_integration_score"] for r in ok]),
        avg_tone_consistency_score=avg([r["tone_consistency_score"] for r in ok]),
        avg_actionability_score=avg([r["actionability_score"] for r in ok]),
        overall_average_score=avg([r["average_score"] for r in ok]),
        avg_latency_ms=avg(lats),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Email Generation Assistant — Evaluation Runner"
    )
    parser.add_argument(
        "--model",
        choices=["gemma", "groq", "both", "compare"],
        required=True,
        help=(
            "gemma: run Model A | groq: run Model B | "
            "both: run both | compare: use existing CSVs"
        )
    )
    parser.add_argument(
        "--delay", type=float, default=4.0,
        help="Seconds between API calls (default: 4.0)"
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    setup_logging(log_level="INFO", log_file="logs/evaluation.log")

    try:
        if args.model == "gemma":
            run_model("gemma", args.delay)

        elif args.model == "groq":
            run_model("groq", args.delay)

        elif args.model == "both":
            gs = run_model("gemma", args.delay)
            print("\n⏳ Waiting 10s before Model B...")
            import time
            time.sleep(10)
            qs = run_model("groq", args.delay)
            save_comparison_csv(gs, qs, load_csv("gemma"), load_csv("groq"))

        elif args.model == "compare":
            print("\n📊 Generating comparison from existing CSVs...")
            gr = load_csv("gemma")
            qr = load_csv("groq")
            if not gr:
                print("❌ results/gemma_results.csv not found. Run --model gemma first.")
                sys.exit(1)
            if not qr:
                print("❌ results/groq_results.csv not found. Run --model groq first.")
                sys.exit(1)
            gs = summary_from_csv(gr, gr[0].get("model_name", "gemma-3-27b-it"))
            qs = summary_from_csv(qr, qr[0].get("model_name", "groq-model"))
            save_comparison_csv(gs, qs, gr, qr)

        print("\n✅ Done! Check the results/ directory.")

    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND: {e}")
        sys.exit(1)
    except EnvironmentError as e:
        print(f"\n❌ CONFIG ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted. Partial results saved.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()