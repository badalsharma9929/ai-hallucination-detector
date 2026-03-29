import requests
import re
from typing import List, Dict, Optional


class FactChecker:
    def __init__(self):
        self.wiki_api = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()

    def extract_claims(self, text: str) -> List[str]:
        claims = []
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+([a-z].+?)(?=\.|$)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+([a-z].+?)(?=\.|$)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+were\s+([a-z].+?)(?=\.|$)',
            r'\bThere\s+are\s+([\d,]+)\s+([a-z\s]+?)(?=\.|$)',
            r'\b([\d,]+)\s+([a-z][\w\s]+?)(?=\.|$)',
        ]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if sent.strip():
                for pattern in patterns:
                    matches = re.findall(pattern, sent, re.IGNORECASE)
                    for match in matches:
                        claims.append({
                            "original": sent,
                            "subject": match[0] if isinstance(match, tuple) else match,
                            "claim": match[1] if isinstance(match, tuple) and len(match) > 1 else match,
                            "verified": None,
                            "wiki_found": None,
                            "confidence": 0.0
                        })
        
        return claims if claims else [{"original": text, "subject": "text", "claim": text, "verified": None, "wiki_found": None, "confidence": 0.0}]

    def search_wikipedia(self, query: str) -> Optional[Dict]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }
        
        try:
            response = self.session.get(self.wiki_api, params=params, timeout=5)
            data = response.json()
            
            if data.get("query") and data["query"]["search"]:
                result = data["query"]["search"][0]
                return {
                    "title": result["title"],
                    "snippet": result["snippet"],
                    "page_id": result["pageid"]
                }
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return None

    def verify_claim(self, claim: str, subject: str = None) -> dict:
        if subject:
            search_query = f"{subject} {claim}"
        else:
            search_query = claim
        
        search_query = search_query[:200]
        wiki_result = self.search_wikipedia(search_query)
        
        verified = False
        confidence = 0.0
        source = None
        
        if wiki_result:
            confidence = 0.7
            source = f"https://en.wikipedia.org/?curid={wiki_result['page_id']}"
            
            claim_lower = claim.lower()
            snippet_lower = wiki_result['snippet'].lower()
            
            key_words = [w for w in claim_lower.split() if len(w) > 4]
            matches = sum(1 for w in key_words if w in snippet_lower)
            
            if matches >= len(key_words) * 0.5:
                verified = True
                confidence = 0.9
        else:
            search_query2 = " ".join(claim.split()[:5])
            wiki_result2 = self.search_wikipedia(search_query2)
            if wiki_result2:
                confidence = 0.4
                source = f"https://en.wikipedia.org/?curid={wiki_result2['page_id']}"

        return {
            "claim": claim,
            "subject": subject,
            "verified": verified,
            "confidence": confidence,
            "source": source,
            "wiki_snippet": wiki_result['snippet'] if wiki_result else None,
            "risk_level": "Low" if verified else ("Medium" if confidence > 0.3 else "High")
        }

    def check_text(self, text: str) -> dict:
        claims = self.extract_claims(text)
        verified_claims = []
        
        for claim in claims:
            result = self.verify_claim(claim["claim"], claim["subject"])
            result["original_sentence"] = claim["original"]
            verified_claims.append(result)
        
        verified_count = sum(1 for c in verified_claims if c["verified"])
        verification_rate = verified_count / len(verified_claims) if verified_claims else 0

        return {
            "text": text,
            "total_claims": len(verified_claims),
            "verified_claims": verified_count,
            "verification_rate": round(verification_rate, 3),
            "risk_level": "Low" if verification_rate > 0.7 else ("Medium" if verification_rate > 0.4 else "High"),
            "claims": verified_claims
        }


if __name__ == "__main__":
    checker = FactChecker()
    test_text = "The capital of France is Paris. The Eiffel Tower is in France."
    result = checker.check_text(test_text)
    print(f"Total Claims: {result['total_claims']}")
    print(f"Verified: {result['verified_claims']}")
    print(f"Verification Rate: {result['verification_rate']}")
    print(f"Risk Level: {result['risk_level']}")
    print("\nClaims:")
    for claim in result['claims']:
        print(f"  - {claim['claim']}: {'Verified' if claim['verified'] else 'Unverified'} (Confidence: {claim['confidence']})")
