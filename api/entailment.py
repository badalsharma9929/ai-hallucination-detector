from transformers import pipeline
import numpy as np


class EntailmentChecker:
    def __init__(self):
        self.nli = pipeline(
            "text-classification",
            model="roberta-base-openai-detector",
            device=-1,
            framework="pt"
        )

    def check_sentence(self, sentence: str, context: str = None) -> dict:
        result = self.nli(sentence)[0]
        
        hallucination_score = 0.0
        if result['label'] == "fake":
            hallucination_score = result['score']
        elif result['label'] == "real":
            hallucination_score = 1.0 - result['score']
        else:
            hallucination_score = 0.5

        return {
            "sentence": sentence,
            "classification": result['label'],
            "confidence": round(result['score'], 3),
            "hallucination_score": round(hallucination_score, 3),
            "risk_level": self._get_risk(hallucination_score)
        }

    def check_text(self, text: str, context: str = None) -> list:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        results = []
        for sent in sentences:
            if sent.strip():
                results.append(self.check_sentence(sent, context))
        return results

    def _get_risk(self, score: float) -> str:
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Medium"
        else:
            return "High"


if __name__ == "__main__":
    checker = EntailmentChecker()
    test_cases = [
        "The Earth is the third planet from the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The moon is made of cheese."
    ]
    
    print("Testing Entailment Checker\n" + "=" * 50)
    for sentence in test_cases:
        result = checker.check_sentence(sentence)
        print(f"Sentence: {result['sentence']}")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Hallucination Score: {result['hallucination_score']}")
        print(f"Risk: {result['risk_level']}")
        print("-" * 50)
