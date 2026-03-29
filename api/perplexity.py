import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np


class PerplexityScorer:
    def __init__(self):
        self.device = "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

    def score_sentence(self, sentence: str) -> dict:
        inputs = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs, labels=inputs)
            loss = outputs.loss.item()
            perplexity = np.exp(loss)
        
        risk_level = "Low"
        if perplexity > 150:
            risk_level = "High"
        elif perplexity > 100:
            risk_level = "Medium"
        
        return {
            "sentence": sentence,
            "perplexity": round(perplexity, 2),
            "risk_level": risk_level,
            "interpretation": self._interpret(perplexity)
        }

    def score_text(self, text: str) -> list:
        sentences = self._split_sentences(text)
        results = []
        for sent in sentences:
            if sent.strip():
                results.append(self.score_sentence(sent))
        return results

    def _split_sentences(self, text: str) -> list:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _interpret(self, perplexity: float) -> str:
        if perplexity < 50:
            return "Very natural, likely human-written"
        elif perplexity < 100:
            return "Normal language model output"
        elif perplexity < 150:
            return "Slightly unusual phrasing detected"
        elif perplexity < 200:
            return "Unusual sentence structure - possible hallucination indicator"
        else:
            return "Highly unusual - strong hallucination indicator"


if __name__ == "__main__":
    scorer = PerplexityScorer()
    test_text = "The capital of France is Paris. The moon is made of green cheese."
    results = scorer.score_text(test_text)
    for r in results:
        print(f"Sentence: {r['sentence']}")
        print(f"Perplexity: {r['perplexity']}")
        print(f"Risk: {r['risk_level']}")
        print(f"Interpretation: {r['interpretation']}")
        print("-" * 50)
