from .perplexity import PerplexityScorer
from .entailment import EntailmentChecker
from .consistency import ConsistencyChecker
from .fact_check import FactChecker


class HallucinationDetector:
    def __init__(self, groq_api_key: str = None):
        self.entailment_checker = EntailmentChecker()
        self.perplexity_scorer = PerplexityScorer()
        self.consistency_checker = ConsistencyChecker(groq_api_key)
        self.fact_checker = FactChecker()

    def analyze(self, text: str, context: str = None) -> dict:
        perplexity_results = self.perplexity_scorer.score_text(text)
        entailment_results = self.entailment_checker.check_text(text, context)
        consistency_results = self.consistency_checker.check_consistency(text)
        fact_check_results = self.fact_checker.check_text(text)

        avg_perplexity = sum(r['perplexity'] for r in perplexity_results) / len(perplexity_results) if perplexity_results else 0
        avg_hallucination = sum(r['hallucination_score'] for r in entailment_results) / len(entailment_results) if entailment_results else 0
        consistency_score = consistency_results.get('consistency_score', 1.0)
        fact_verification_rate = fact_check_results.get('verification_rate', 0)

        overall_score = (
            (min(avg_perplexity / 200, 1.0) * 0.25) +
            (avg_hallucination * 0.25) +
            ((1 - consistency_score) * 0.25) +
            ((1 - fact_verification_rate) * 0.25)
        )

        risk_level = "Low"
        if overall_score > 0.5:
            risk_level = "High"
        elif overall_score > 0.3:
            risk_level = "Medium"

        flagged_sentences = []
        for r in perplexity_results:
            if r['risk_level'] in ['Medium', 'High']:
                flagged_sentences.append({
                    "sentence": r['sentence'],
                    "reason": f"High perplexity ({r['perplexity']})",
                    "risk_level": r['risk_level']
                })

        for r in entailment_results:
            if r['risk_level'] in ['Medium', 'High']:
                flagged_sentences.append({
                    "sentence": r['sentence'],
                    "reason": f"NLI classification: {r['classification']}",
                    "risk_level": r['risk_level']
                })

        for r in fact_check_results.get('claims', []):
            if r['risk_level'] in ['Medium', 'High']:
                flagged_sentences.append({
                    "sentence": r['original_sentence'],
                    "reason": "Fact not verified via Wikipedia",
                    "risk_level": r['risk_level']
                })

        return {
            "text": text,
            "overall_score": round(overall_score, 3),
            "risk_level": risk_level,
            "metrics": {
                "perplexity": {
                    "average": round(avg_perplexity, 2),
                    "interpretation": self._interpret_perplexity(avg_perplexity)
                },
                "entailment": {
                    "average_hallucination_score": round(avg_hallucination, 3),
                    "interpretation": self._interpret_entailment(avg_hallucination)
                },
                "consistency": {
                    "score": round(consistency_score, 3),
                    "interpretation": self._interpret_consistency(consistency_score)
                },
                "fact_verification": {
                    "verification_rate": round(fact_verification_rate, 3),
                    "interpretation": self._interpret_fact_check(fact_verification_rate)
                }
            },
            "flagged_sentences": flagged_sentences,
            "recommendations": self._generate_recommendations(overall_score, risk_level)
        }

    def _interpret_perplexity(self, score: float) -> str:
        if score < 50:
            return "Very natural language"
        elif score < 100:
            return "Normal AI-generated text"
        elif score < 150:
            return "Slightly unusual phrasing"
        else:
            return "Highly unusual - possible hallucination"

    def _interpret_entailment(self, score: float) -> str:
        if score < 0.2:
            return "Claims appear factual"
        elif score < 0.5:
            return "Some questionable claims detected"
        else:
            return "Multiple false claims likely"

    def _interpret_consistency(self, score: float) -> str:
        if score > 0.7:
            return "High internal consistency"
        elif score > 0.4:
            return "Some inconsistent claims"
        else:
            return "Low consistency - possible fabrication"

    def _interpret_fact_check(self, rate: float) -> str:
        if rate > 0.7:
            return "Most facts verifiable"
        elif rate > 0.4:
            return "Some facts unverifiable"
        else:
            return "Most facts not verified"

    def _generate_recommendations(self, score: float, risk: str) -> list:
        recommendations = []
        if risk == "High":
            recommendations.append("Review flagged sentences carefully before using this text.")
            recommendations.append("Cross-reference claims with authoritative sources.")
            recommendations.append("Consider rewriting high-risk sections.")
        elif risk == "Medium":
            recommendations.append("Verify uncertain claims with additional sources.")
            recommendations.append("Review perplexity-flagged sentences.")
        else:
            recommendations.append("Text appears generally reliable.")
            recommendations.append("Always verify specific facts when possible.")
        return recommendations
