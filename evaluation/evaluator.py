"""
Evaluation Pipeline — Orchestrator.

Loads 10 scenarios, generates emails, scores on 3 metrics,
saves CSV + JSON. Partial results saved after every scenario.
"""

import json
import csv
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logger import get_logger, ContextLogger
from config.settings import load_config
from models.gemini_client import GeminiClient
from models.groq_client import GroqClient
from models.base_client import EmailGenerationRequest
from evaluation.metrics import (
    compute_fact_integration_score,
    compute_tone_consistency_score,
    compute_actionability_score,
    MetricResult,
)

logger = get_logger(__name__)


@dataclass
class ScenarioResult:
    """Complete evaluation result for one scenario."""
    scenario_id: int
    intent: str
    tone: str
    key_facts_count: int
    model_name: str
    generated_email: str
    generation_success: bool
    generation_latency_ms: Optional[float]
    generation_error: Optional[str]
    fact_integration_score: float = 0.0
    tone_consistency_score: float = 0.0
    actionability_score: float = 0.0
    average_score: float = 0.0
    fact_integration_reasoning: str = ""
    tone_consistency_reasoning: str = ""
    actionability_reasoning: str = ""
    fact_integration_error: Optional[str] = None
    tone_consistency_error: Optional[str] = None
    actionability_error: Optional[str] = None
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EvaluationSummary:
    """Aggregated summary across all 10 scenarios."""
    model_name: str
    total_scenarios: int
    successful_generations: int
    failed_generations: int
    avg_fact_integration_score: float
    avg_tone_consistency_score: float
    avg_actionability_score: float
    overall_average_score: float
    avg_latency_ms: float
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EvaluationPipeline:
    """
    Main evaluation orchestrator.

    Usage:
        pipeline = EvaluationPipeline(model_name="gemma")
        summary = pipeline.run()
    """

    METRIC_DEFINITIONS = {
        "fact_integration_score": (
            "Pure Python keyword matching. "
            "Score = key_facts_found / total_key_facts."
        ),
        "tone_consistency_score": (
            "LLM-as-Judge scores opening/body/closing sections "
            "independently (1-10). Final = avg/10 minus drift penalty."
        ),
        "actionability_score": (
            "LLM-as-Judge checks 3 components: clear ask, next step, "
            "timeframe. Score = components_present / 3."
        ),
    }

    def __init__(
        self,
        model_name: str,
        rate_limit_delay: float = 4.0,
        scenarios_path: str = "scenarios/test_scenarios.json",
        results_dir: str = "results",
    ):
        if model_name not in ("gemma", "groq"):
            raise ValueError(f"model_name must be 'gemma' or 'groq', got '{model_name}'")

        self.model_name = model_name
        self.rate_limit_delay = rate_limit_delay
        self.scenarios_path = scenarios_path
        self.results_dir = results_dir
        self._config = load_config()
        self._llm_client = self._init_client()
        self._results: List[ScenarioResult] = []
        os.makedirs(results_dir, exist_ok=True)

        logger.info(
            "EvaluationPipeline initialized",
            extra={"extra_fields": {"model": model_name}}
        )

    def _init_client(self):
        if self.model_name == "gemma":
            return GeminiClient(self._config.gemini)
        return GroqClient(self._config.groq)

    def _judge_fn(self, prompt: str) -> str:
        """
        LLM-as-Judge: calls same model with scoring prompt.
        Metric 1 does NOT use this — it's pure Python.
        Metrics 2 and 3 use this.
        """
        request = EmailGenerationRequest(
            intent="judge", key_facts=["judge"],
            tone="formal", correlation_id="judge",
        )
        response = self._llm_client.generate_email(prompt, request)
        if not response.success:
            raise RuntimeError(f"Judge call failed: {response.error_message}")
        return response.email_text

    def run(self) -> EvaluationSummary:
        """Run the full evaluation pipeline."""
        run_id = str(uuid.uuid4())[:8]
        ctx_logger = ContextLogger(logger, run_id)

        ctx_logger.info(f"Starting evaluation — model: {self.model_name}")

        scenarios = self._load_scenarios()
        ctx_logger.info(f"Loaded {len(scenarios)} scenarios")

        if not self._llm_client.health_check():
            ctx_logger.warning("Health check failed — proceeding anyway")

        self._results = []
        total = len(scenarios)

        for i, scenario in enumerate(scenarios, 1):
            sid = scenario.get("id", i)
            ctx_logger.info(
                f"[{i}/{total}] Scenario {sid}: {scenario.get('intent','')[:50]}"
            )

            result = self._evaluate_scenario(scenario)
            self._results.append(result)
            self._save_partial(run_id)

            ctx_logger.info(
                f"Scenario {sid} done — "
                f"avg:{result.average_score:.2f} "
                f"fact:{result.fact_integration_score:.2f} "
                f"tone:{result.tone_consistency_score:.2f} "
                f"action:{result.actionability_score:.2f}"
            )

            if i < total:
                time.sleep(self.rate_limit_delay)

        csv_path = self._save_csv()
        summary = self._build_summary()
        self._print_table(summary)

        ctx_logger.info(
            "Evaluation complete",
            extra={"extra_fields": {
                "overall": summary.overall_average_score,
                "csv": csv_path,
            }}
        )

        return summary

    def _evaluate_scenario(self, scenario: Dict) -> ScenarioResult:
        """Generate email + score on 3 metrics for one scenario."""
        sid = scenario.get("id", 0)
        intent = scenario.get("intent", "")
        key_facts = scenario.get("key_facts", [])
        tone = scenario.get("tone", "formal")
        corr_id = f"s{sid}-{str(uuid.uuid4())[:4]}"

        if not intent:
            return self._fail(sid, intent, tone, key_facts, "Missing intent")
        if not key_facts:
            return self._fail(sid, intent, tone, key_facts, "Missing key_facts")

        # Build prompt
        from prompts.advanced_prompt import build_prompt
        try:
            prompt = build_prompt(intent, key_facts, tone)
        except ValueError as e:
            return self._fail(sid, intent, tone, key_facts, str(e))

        # Generate email
        request = EmailGenerationRequest(
            intent=intent, key_facts=key_facts,
            tone=tone, correlation_id=corr_id,
        )
        gen = self._llm_client.generate_email(prompt, request)

        if not gen.success:
            return self._fail(sid, intent, tone, key_facts,
                              gen.error_message or "Generation failed")

        email = gen.email_text

        # ── Metric 1: Fact Integration (pure Python — NO judge_fn) ──
        fact: MetricResult = compute_fact_integration_score(
            generated_email=email,
            key_facts=key_facts,
            correlation_id=corr_id,
        )

        time.sleep(1.5)

        # ── Metric 2: Tone Consistency (LLM-as-Judge) ──
        tone_r: MetricResult = compute_tone_consistency_score(
            generated_email=email,
            expected_tone=tone,
            judge_fn=self._judge_fn,
            correlation_id=corr_id,
        )

        time.sleep(1.5)

        # ── Metric 3: Actionability (LLM-as-Judge) ──
        action: MetricResult = compute_actionability_score(
            generated_email=email,
            intent=intent,
            judge_fn=self._judge_fn,
            correlation_id=corr_id,
        )

        avg = round((fact.score + tone_r.score + action.score) / 3, 4)

        return ScenarioResult(
            scenario_id=sid,
            intent=intent,
            tone=tone,
            key_facts_count=len(key_facts),
            model_name=self._llm_client.model_name,
            generated_email=email,
            generation_success=True,
            generation_latency_ms=gen.latency_ms,
            generation_error=None,
            fact_integration_score=fact.score,
            tone_consistency_score=tone_r.score,
            actionability_score=action.score,
            average_score=avg,
            fact_integration_reasoning=fact.reasoning,
            tone_consistency_reasoning=tone_r.reasoning,
            actionability_reasoning=action.reasoning,
            fact_integration_error=fact.error,
            tone_consistency_error=tone_r.error,
            actionability_error=action.error,
        )

    def _fail(self, sid, intent, tone, key_facts, error) -> ScenarioResult:
        logger.error(f"Scenario {sid} failed: {error}")
        return ScenarioResult(
            scenario_id=sid, intent=intent, tone=tone,
            key_facts_count=len(key_facts) if key_facts else 0,
            model_name=self._llm_client.model_name,
            generated_email="", generation_success=False,
            generation_latency_ms=None, generation_error=error,
        )

    def _load_scenarios(self) -> List[Dict]:
        if not os.path.exists(self.scenarios_path):
            raise FileNotFoundError(
                f"Scenarios file not found: '{self.scenarios_path}'\n"
                "Make sure scenarios/test_scenarios.json exists."
            )
        with open(self.scenarios_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Scenarios file must be a non-empty JSON array")
        return data

    def _save_partial(self, run_id: str) -> None:
        path = os.path.join(
            self.results_dir, f"{self.model_name}_partial_{run_id}.csv"
        )
        fields = ["scenario_id","intent","tone","model_name",
                  "generation_success","generation_latency_ms",
                  "fact_integration_score","tone_consistency_score",
                  "actionability_score","average_score","evaluated_at"]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in self._results:
                    w.writerow(asdict(r))
        except Exception as e:
            logger.warning(f"Could not save partial results: {e}")

    def _save_csv(self) -> str:
        path = os.path.join(self.results_dir, f"{self.model_name}_results.csv")
        fields = [
            "scenario_id", "intent", "tone", "key_facts_count", "model_name",
            "generation_success", "generation_latency_ms", "generation_error",
            "fact_integration_score", "tone_consistency_score",
            "actionability_score", "average_score",
            "fact_integration_reasoning", "tone_consistency_reasoning",
            "actionability_reasoning",
            "fact_integration_error", "tone_consistency_error",
            "actionability_error", "evaluated_at",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in self._results:
                    w.writerow(asdict(r))
            logger.info(f"CSV saved: {path}")
        except Exception as e:
            logger.error(f"CSV save failed: {e}")
        return path



    def _build_summary(self) -> EvaluationSummary:
        ok = [r for r in self._results if r.generation_success]
        fail = [r for r in self._results if not r.generation_success]

        def avg(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        lats = [r.generation_latency_ms for r in ok if r.generation_latency_ms]

        return EvaluationSummary(
            model_name=self._llm_client.model_name,
            total_scenarios=len(self._results),
            successful_generations=len(ok),
            failed_generations=len(fail),
            avg_fact_integration_score=avg([r.fact_integration_score for r in ok]),
            avg_tone_consistency_score=avg([r.tone_consistency_score for r in ok]),
            avg_actionability_score=avg([r.actionability_score for r in ok]),
            overall_average_score=avg([r.average_score for r in ok]),
            avg_latency_ms=avg(lats),
        )

    def _print_table(self, s: EvaluationSummary) -> None:
        print("\n" + "=" * 65)
        print(f"  EVALUATION SUMMARY — {s.model_name.upper()}")
        print("=" * 65)
        print(f"  Total     : {s.total_scenarios}")
        print(f"  Successful: {s.successful_generations}  Failed: {s.failed_generations}")
        print("-" * 65)
        print(f"  Fact Integration : {s.avg_fact_integration_score:.4f}")
        print(f"  Tone Consistency : {s.avg_tone_consistency_score:.4f}")
        print(f"  Actionability    : {s.avg_actionability_score:.4f}")
        print("-" * 65)
        print(f"  OVERALL AVERAGE  : {s.overall_average_score:.4f}")
        print(f"  Avg Latency      : {s.avg_latency_ms:.0f}ms")
        print("=" * 65)
        print(f"\n  {'ID':<4} {'Tone':<12} {'Fact':>6} {'Tone':>6} {'Act':>6} {'Avg':>6}")
        print("  " + "-" * 45)
        for r in self._results:
            ok = "✅" if r.generation_success else "❌"
            print(
                f"  {r.scenario_id:<4} {r.tone:<12} "
                f"{r.fact_integration_score:>6.2f} "
                f"{r.tone_consistency_score:>6.2f} "
                f"{r.actionability_score:>6.2f} "
                f"{r.average_score:>6.2f} {ok}"
            )
        print()