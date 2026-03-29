# AI Hallucination Detector

A production-ready tool to detect potential hallucinations in AI-generated text using multiple detection methods.

## Features

- **Perplexity Analysis** - Measures how unusual/unnatural the text sounds using GPT-2
- **AI Text Detection** - Detects AI-generated text using RoBERTa (OpenAI detector)
- **Self-Consistency** - Verifies internal consistency by generating multiple responses (Groq API)
- **Fact Verification** - Cross-references facts with Wikipedia API

## Tech Stack

- **Perplexity**: GPT-2 (HuggingFace, free, CPU)
- **AI Text Detection**: roberta-base-openai-detector (HuggingFace, free, CPU)
- **LLM**: Groq API (Llama-3.1-8B-Instant, free tier)
- **Fact Check**: Wikipedia API (free)
- **Frontend**: Streamlit
- **Backend**: FastAPI ready

## Installation

```bash
# Clone the repository
git clone https://github.com/badalsharma9929/ai-hallucination-detector.git
cd ai-hallucination-detector

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running locally

```bash
# Start the Streamlit app
streamlit run app.py
```

### Optional: Set Groq API Key

For self-consistency checks, set your Groq API key:

```bash
export GROQ_API_KEY="your_api_key_here"
```

Or enter it in the Streamlit UI.

## Getting Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Use it in the app (free tier: 30 requests/minute)

## Project Structure

```
ai-hallucination-detector/
├── app.py                    # Streamlit UI
├── api/
│   ├── __init__.py
│   ├── detector.py           # Main orchestration
│   ├── perplexity.py         # Perplexity scoring (GPT-2)
│   ├── entailment.py         # AI text detection (RoBERTa)
│   ├── consistency.py        # Self-consistency (Groq)
│   └── fact_check.py         # Wikipedia verification
├── requirements.txt
└── README.md
```

## How It Works

1. **Input**: Paste any text to analyze
2. **Perplexity**: GPT-2 scores how "natural" each sentence sounds
3. **AI Detection**: RoBERTa detects if text is AI-generated
4. **Consistency**: Multiple LLM generations are compared for contradictions
5. **Fact Check**: Claims are verified against Wikipedia
6. **Output**: Overall hallucination score + flagged sentences

## Deployment

Deploy on Streamlit Cloud for free:

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Deploy!

## Example Output

```
Overall Hallucination Score: 23.5%
Risk Level: Low

Flagged Sentences:
🔴 High Risk: "The moon is made of green cheese"
   Reason: Fact not verified via Wikipedia

Metrics:
- Perplexity: 145.23 (Slightly unusual)
- Entailment: 0.67 hallucination score
- Consistency: 0.85
- Fact Verification: 0.45
```

## Contributing

Contributions welcome! Feel free to open issues and PRs.

## License

MIT License
