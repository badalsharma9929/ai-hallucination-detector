import os
import re
from groq import Groq


class ConsistencyChecker:
    def __init__(self, groq_api_key: str = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: GROQ_API_KEY not set. Self-consistency checks will use fallback.")

    def generate_with_llm(self, prompt: str, temperature: float = 0.7) -> str:
        if not self.client:
            return "API key required for LLM generation"
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions directly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def extract_facts(self, text: str) -> list:
        facts = []
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+is\s+',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+was\s+',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+were\s+',
            r'\bThere\s+are\s+\d+',
            r'\bIt\s+takes\s+\d+',
        ]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if sent.strip():
                facts.append(sent.strip())
        return facts

    def check_consistency(self, text: str, num_generations: int = 3) -> dict:
        facts = self.extract_facts(text)
        
        if not facts:
            return {
                "text": text,
                "consistency_score": 1.0,
                "risk_level": "Low",
                "inconsistent_claims": [],
                "generations": []
            }

        generations = []
        inconsistencies = []

        if self.client:
            for i in range(num_generations):
                prompt = f"Based on the following claims, answer what you know to be true: {' '.join(facts)}"
                gen_text = self.generate_with_llm(prompt, temperature=0.7 + (i * 0.1))
                generations.append(gen_text)
            
            for fact in facts:
                matches = sum(1 for gen in generations if any(
                    word in gen.lower() for word in fact.lower().split()[:3]
                ))
                consistency = matches / len(generations) if generations else 0
                if consistency < 0.5:
                    inconsistencies.append({
                        "claim": fact,
                        "consistency": consistency,
                        "supporting_generations": matches
                    })
        else:
            generations = ["LLM not configured - set GROQ_API_KEY"]

        consistency_score = 1.0 - (len(inconsistencies) / len(facts)) if facts else 1.0

        return {
            "text": text,
            "facts_extracted": len(facts),
            "consistency_score": round(max(0, consistency_score), 3),
            "risk_level": "Low" if consistency_score > 0.7 else ("Medium" if consistency_score > 0.4 else "High"),
            "inconsistent_claims": inconsistencies,
            "generations": generations[:3]
        }


if __name__ == "__main__":
    checker = ConsistencyChecker()
    test_text = "The capital of France is Paris. It has over 2 million residents."
    result = checker.check_consistency(test_text)
    print(f"Consistency Score: {result['consistency_score']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Facts Extracted: {result['facts_extracted']}")
    if result['inconsistent_claims']:
        print("Inconsistent Claims:")
        for claim in result['inconsistent_claims']:
            print(f"  - {claim['claim']} (consistency: {claim['consistency']})")
