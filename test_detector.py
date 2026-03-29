import sys
sys.path.insert(0, '.')
from api.detector import HallucinationDetector

print('Loading HallucinationDetector...')
d = HallucinationDetector()
print('Loaded!')
result = d.analyze('The capital of France is Paris. The moon is made of cheese.')
print(f'Score: {result["overall_score"]:.1%}, Risk: {result["risk_level"]}')
print(f'Perplexity avg: {result["metrics"]["perplexity"]["average"]}')
print(f'Entailment avg: {result["metrics"]["entailment"]["average_hallucination_score"]}')
print(f'Fact verify rate: {result["metrics"]["fact_verification"]["verification_rate"]}')
print(f'Flagged: {len(result["flagged_sentences"])} sentences')
for f in result['flagged_sentences']:
    print(f'  - [{f["risk_level"]}] {f["sentence"]}')
print('All OK!')
