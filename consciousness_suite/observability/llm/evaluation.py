"""LLM Evaluation Module

Provides evaluation metrics and scoring for LLM outputs:
- Response quality metrics
- Hallucination detection
- RAG evaluation (relevance, faithfulness)
- Custom evaluation criteria
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class EvaluationMetric(str, Enum):
    """Standard evaluation metrics."""
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    GROUNDEDNESS = "groundedness"
    ANSWER_CORRECTNESS = "answer_correctness"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    metric: str
    score: float  # 0-1 normalized score
    passed: bool
    confidence: float
    reasoning: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    overall_score: float
    results: List[EvaluationResult]
    passed: bool
    summary: str
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    context: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMEvaluator:
    """LLM output evaluator.

    Usage:
        evaluator = LLMEvaluator()

        # Evaluate a response
        result = evaluator.evaluate(
            input_text="What is the capital of France?",
            output_text="The capital of France is Paris.",
            metrics=[EvaluationMetric.RELEVANCE, EvaluationMetric.COHERENCE],
        )

        # RAG evaluation
        rag_result = evaluator.evaluate_rag(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language..."],
        )
    """

    def __init__(
        self,
        default_threshold: float = 0.7,
        use_llm_judge: bool = False,
        llm_judge_model: Optional[str] = None,
    ):
        self.default_threshold = default_threshold
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model or "gpt-4o-mini"

        # Custom evaluators
        self._custom_evaluators: Dict[str, Callable] = {}

        # Metric weights for overall score
        self._metric_weights: Dict[str, float] = {
            EvaluationMetric.RELEVANCE: 1.0,
            EvaluationMetric.FAITHFULNESS: 1.0,
            EvaluationMetric.COHERENCE: 0.8,
            EvaluationMetric.FLUENCY: 0.6,
            EvaluationMetric.HELPFULNESS: 0.9,
            EvaluationMetric.HARMLESSNESS: 1.2,
        }

    def register_evaluator(
        self,
        name: str,
        evaluator: Callable[[str, str, Optional[List[str]]], EvaluationResult],
        weight: float = 1.0,
    ):
        """Register a custom evaluator.

        Args:
            name: Metric name
            evaluator: Function that takes (input, output, context) and returns EvaluationResult
            weight: Weight for overall score calculation
        """
        self._custom_evaluators[name] = evaluator
        self._metric_weights[name] = weight

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        metrics: Optional[List[Union[EvaluationMetric, str]]] = None,
        context: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> EvaluationReport:
        """Evaluate an LLM output.

        Args:
            input_text: The input/prompt
            output_text: The LLM output to evaluate
            metrics: List of metrics to evaluate
            context: Optional context documents (for RAG)
            threshold: Pass/fail threshold (default: 0.7)

        Returns:
            EvaluationReport with scores
        """
        if metrics is None:
            metrics = [EvaluationMetric.RELEVANCE, EvaluationMetric.COHERENCE]

        threshold = threshold or self.default_threshold
        results = []

        for metric in metrics:
            metric_name = metric.value if isinstance(metric, EvaluationMetric) else metric

            # Try custom evaluator first
            if metric_name in self._custom_evaluators:
                result = self._custom_evaluators[metric_name](input_text, output_text, context)
            else:
                result = self._evaluate_metric(metric_name, input_text, output_text, context)

            result.passed = result.score >= threshold
            results.append(result)

        # Calculate overall score
        total_weight = sum(
            self._metric_weights.get(r.metric, 1.0) for r in results
        )
        weighted_sum = sum(
            r.score * self._metric_weights.get(r.metric, 1.0) for r in results
        )
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        passed = overall_score >= threshold and all(r.passed for r in results)

        # Generate summary
        failed_metrics = [r.metric for r in results if not r.passed]
        if passed:
            summary = f"Evaluation passed with overall score {overall_score:.2f}"
        else:
            summary = f"Evaluation failed. Failed metrics: {', '.join(failed_metrics)}"

        return EvaluationReport(
            overall_score=overall_score,
            results=results,
            passed=passed,
            summary=summary,
            input_text=input_text,
            output_text=output_text,
            context=context,
        )

    def _evaluate_metric(
        self,
        metric: str,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate a single metric."""
        evaluators = {
            EvaluationMetric.RELEVANCE.value: self._evaluate_relevance,
            EvaluationMetric.COHERENCE.value: self._evaluate_coherence,
            EvaluationMetric.FLUENCY.value: self._evaluate_fluency,
            EvaluationMetric.FAITHFULNESS.value: self._evaluate_faithfulness,
            EvaluationMetric.HARMLESSNESS.value: self._evaluate_harmlessness,
            EvaluationMetric.HELPFULNESS.value: self._evaluate_helpfulness,
        }

        if metric in evaluators:
            return evaluators[metric](input_text, output_text, context)

        # Unknown metric - return neutral score
        return EvaluationResult(
            metric=metric,
            score=0.5,
            passed=True,
            confidence=0.0,
            reasoning="Unknown metric - no evaluation performed",
        )

    def _evaluate_relevance(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate response relevance to the input."""
        # Simple heuristic-based evaluation
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "and",
                      "or", "but", "if", "then", "else", "when", "where", "why",
                      "how", "what", "which", "who", "whom", "this", "that", "these",
                      "those", "i", "you", "he", "she", "it", "we", "they", "to",
                      "of", "in", "for", "on", "with", "at", "by", "from", "as"}

        input_keywords = input_words - stop_words
        output_keywords = output_words - stop_words

        if not input_keywords:
            return EvaluationResult(
                metric=EvaluationMetric.RELEVANCE.value,
                score=0.5,
                passed=True,
                confidence=0.3,
                reasoning="Input has no significant keywords",
            )

        # Calculate keyword overlap
        overlap = len(input_keywords & output_keywords)
        overlap_ratio = overlap / len(input_keywords)

        # Boost score if output is substantial
        length_factor = min(len(output_text) / 50, 1.0)

        score = min((overlap_ratio * 0.7 + length_factor * 0.3), 1.0)

        return EvaluationResult(
            metric=EvaluationMetric.RELEVANCE.value,
            score=score,
            passed=score >= self.default_threshold,
            confidence=0.7,
            reasoning=f"Keyword overlap: {overlap}/{len(input_keywords)}",
            details={"overlap_ratio": overlap_ratio, "length_factor": length_factor},
        )

    def _evaluate_coherence(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate response coherence."""
        # Check for coherence indicators
        sentences = re.split(r'[.!?]+', output_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return EvaluationResult(
                metric=EvaluationMetric.COHERENCE.value,
                score=0.0,
                passed=False,
                confidence=0.9,
                reasoning="No complete sentences found",
            )

        # Check sentence structure
        well_formed = sum(
            1 for s in sentences
            if len(s.split()) >= 3 and s[0].isupper()
        )
        structure_score = well_formed / len(sentences) if sentences else 0

        # Check for coherence markers
        coherence_markers = [
            "therefore", "however", "furthermore", "additionally",
            "consequently", "moreover", "thus", "hence", "because",
            "since", "although", "while", "first", "second", "finally",
        ]
        marker_count = sum(
            1 for marker in coherence_markers
            if marker in output_text.lower()
        )
        marker_score = min(marker_count / 3, 1.0)

        # Overall coherence score
        score = structure_score * 0.7 + marker_score * 0.3

        return EvaluationResult(
            metric=EvaluationMetric.COHERENCE.value,
            score=score,
            passed=score >= self.default_threshold,
            confidence=0.6,
            reasoning=f"Structure score: {structure_score:.2f}, Marker score: {marker_score:.2f}",
        )

    def _evaluate_fluency(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate language fluency."""
        # Check for basic fluency indicators
        words = output_text.split()

        if not words:
            return EvaluationResult(
                metric=EvaluationMetric.FLUENCY.value,
                score=0.0,
                passed=False,
                confidence=0.9,
                reasoning="Empty output",
            )

        # Average word length (good indicator of vocabulary)
        avg_word_len = sum(len(w) for w in words) / len(words)
        vocab_score = min(avg_word_len / 6, 1.0)

        # Check for repeated words (sign of poor fluency)
        word_freq = {}
        for w in words:
            word_freq[w.lower()] = word_freq.get(w.lower(), 0) + 1

        max_repeat = max(word_freq.values()) if word_freq else 0
        repeat_penalty = max(0, 1 - (max_repeat - 3) * 0.1)

        # Check for proper punctuation
        punct_count = sum(1 for c in output_text if c in ".,!?;:")
        punct_score = min(punct_count / (len(words) / 10 + 1), 1.0)

        score = vocab_score * 0.4 + repeat_penalty * 0.3 + punct_score * 0.3

        return EvaluationResult(
            metric=EvaluationMetric.FLUENCY.value,
            score=score,
            passed=score >= self.default_threshold,
            confidence=0.5,
            reasoning=f"Vocab: {vocab_score:.2f}, Repeat: {repeat_penalty:.2f}, Punct: {punct_score:.2f}",
        )

    def _evaluate_faithfulness(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate faithfulness to provided context (for RAG)."""
        if not context:
            return EvaluationResult(
                metric=EvaluationMetric.FAITHFULNESS.value,
                score=1.0,
                passed=True,
                confidence=0.3,
                reasoning="No context provided - cannot evaluate faithfulness",
            )

        # Combine context
        context_text = " ".join(context).lower()
        output_lower = output_text.lower()

        # Extract key claims from output (simplified)
        output_sentences = re.split(r'[.!?]+', output_text)
        output_sentences = [s.strip() for s in output_sentences if s.strip()]

        if not output_sentences:
            return EvaluationResult(
                metric=EvaluationMetric.FAITHFULNESS.value,
                score=0.5,
                passed=True,
                confidence=0.3,
                reasoning="No claims found in output",
            )

        # Check if key words from output are in context
        supported_sentences = 0
        for sentence in output_sentences:
            words = set(sentence.lower().split())
            content_words = words - {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "on", "with"}

            if not content_words:
                supported_sentences += 1
                continue

            matches = sum(1 for w in content_words if w in context_text)
            if matches >= len(content_words) * 0.3:
                supported_sentences += 1

        score = supported_sentences / len(output_sentences)

        return EvaluationResult(
            metric=EvaluationMetric.FAITHFULNESS.value,
            score=score,
            passed=score >= self.default_threshold,
            confidence=0.6,
            reasoning=f"{supported_sentences}/{len(output_sentences)} sentences supported by context",
        )

    def _evaluate_harmlessness(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate for potentially harmful content."""
        # Simple keyword-based check (would use more sophisticated methods in production)
        harmful_patterns = [
            r"\b(kill|murder|attack|weapon|bomb|explosive)\b",
            r"\b(hack|crack|exploit|malware|virus)\b",
            r"\b(hate|racist|sexist|discriminat)\b",
            r"\b(illegal|criminal|fraud)\b",
        ]

        output_lower = output_text.lower()
        harmful_matches = 0

        for pattern in harmful_patterns:
            if re.search(pattern, output_lower):
                harmful_matches += 1

        # Score is inverse of harmful matches
        score = max(0, 1 - harmful_matches * 0.25)

        return EvaluationResult(
            metric=EvaluationMetric.HARMLESSNESS.value,
            score=score,
            passed=score >= 0.75,
            confidence=0.7,
            reasoning=f"Harmful pattern matches: {harmful_matches}",
        )

    def _evaluate_helpfulness(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate response helpfulness."""
        # Check for direct answer indicators
        answer_indicators = [
            "here is", "here's", "the answer is", "you can", "to do this",
            "steps:", "first,", "1.", "2.", "the solution",
        ]

        output_lower = output_text.lower()
        indicator_count = sum(1 for i in answer_indicators if i in output_lower)

        # Check response length (helpful responses tend to be substantial)
        length_score = min(len(output_text) / 200, 1.0)

        # Check for action items or clear guidance
        has_structure = any([
            re.search(r'\d+\.', output_text),  # numbered lists
            re.search(r'[-*]', output_text),   # bullet points
            "step" in output_lower,
            "example" in output_lower,
        ])

        indicator_score = min(indicator_count / 2, 1.0)
        structure_bonus = 0.2 if has_structure else 0

        score = min(indicator_score * 0.4 + length_score * 0.4 + structure_bonus, 1.0)

        return EvaluationResult(
            metric=EvaluationMetric.HELPFULNESS.value,
            score=score,
            passed=score >= self.default_threshold,
            confidence=0.5,
            reasoning=f"Indicators: {indicator_count}, Length: {length_score:.2f}, Structured: {has_structure}",
        )

    def evaluate_rag(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        threshold: float = 0.7,
    ) -> EvaluationReport:
        """Evaluate a RAG (Retrieval Augmented Generation) response.

        Args:
            question: The user question
            answer: The generated answer
            contexts: Retrieved context documents
            threshold: Pass/fail threshold

        Returns:
            EvaluationReport with RAG-specific metrics
        """
        metrics = [
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.FAITHFULNESS,
            EvaluationMetric.COHERENCE,
            EvaluationMetric.HELPFULNESS,
        ]

        return self.evaluate(
            input_text=question,
            output_text=answer,
            metrics=metrics,
            context=contexts,
            threshold=threshold,
        )
