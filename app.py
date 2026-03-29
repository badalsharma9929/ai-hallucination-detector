import streamlit as st
import time
import os
from api.detector import HallucinationDetector


@st.cache_resource
def load_detector(groq_api_key):
    return HallucinationDetector(groq_api_key=groq_api_key)


st.set_page_config(
    page_title="AI Hallucination Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Hallucination Detector")
st.markdown("*Detect potential hallucinations in AI-generated text using multiple detection methods*")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Text")
    text_input = st.text_area(
        "Paste the text you want to analyze:",
        height=200,
        placeholder="Enter any text to check for potential hallucinations..."
    )
    
    context_input = st.text_area(
        "Context (optional):",
        height=100,
        placeholder="Provide context if available (e.g., source document, reference material)..."
    )
    
    analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

with col2:
    st.subheader("Detection Methods")
    st.markdown("""
    - **Perplexity Analysis**: Measures how unusual the text sounds using GPT-2
    - **NLI Entailment**: Checks if claims are factual, questionable, or false
    - **Self-Consistency**: Verifies internal consistency by generating multiple responses
    - **Fact Verification**: Cross-references facts with Wikipedia
    """)
    
    st.subheader("API Key (Optional)")
    st.markdown("For self-consistency checks, add your Groq API key:")
    groq_key = st.text_input("GROQ_API_KEY", type="password", label_visibility="collapsed")

if analyze_button and text_input:
    detector = load_detector(groq_key if groq_key else None)
    
    with st.spinner("Analyzing text..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Running perplexity analysis...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("Running NLI entailment checks...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        status_text.text("Checking self-consistency...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        status_text.text("Verifying facts with Wikipedia...")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        result = detector.analyze(text_input, context_input if context_input else None)
        
        progress_bar.empty()
        status_text.empty()

    st.divider()
    
    col_score, col_risk = st.columns(2)
    
    with col_score:
        score_value = result['overall_score']
        score_display = f"{score_value:.1%}"
        st.metric("Overall Hallucination Score", score_display)
    
    with col_risk:
        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(result['risk_level'], "gray")
        st.markdown(f"**Risk Level:** :{risk_color}[{result['risk_level']}]")
    
    if result['flagged_sentences']:
        st.subheader("🚨 Flagged Sentences")
        for i, flagged in enumerate(result['flagged_sentences'][:10], 1):
            risk_emoji = "🔴" if flagged['risk_level'] == 'High' else "🟡"
            st.markdown(f"{risk_emoji} *{flagged['risk_level']} Risk*: {flagged['sentence']}")
            st.caption(f"Reason: {flagged['reason']}")
            st.divider()
    else:
        st.success("✅ No significant hallucination indicators found!")
    
    with st.expander("📊 Detailed Metrics"):
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Perplexity Analysis**")
            perp = result['metrics']['perplexity']
            st.write(f"- Average Score: {perp['average']}")
            st.write(f"- Interpretation: {perp['interpretation']}")
            
            st.markdown("**Entailment Check**")
            ent = result['metrics']['entailment']
            st.write(f"- Hallucination Score: {ent['average_hallucination_score']}")
            st.write(f"- Interpretation: {ent['interpretation']}")
        
        with col_m2:
            st.markdown("**Self-Consistency**")
            cons = result['metrics']['consistency']
            st.write(f"- Consistency Score: {cons['score']}")
            st.write(f"- Interpretation: {cons['interpretation']}")
            
            st.markdown("**Fact Verification**")
            fact = result['metrics']['fact_verification']
            st.write(f"- Verification Rate: {fact['verification_rate']}")
            st.write(f"- Interpretation: {fact['interpretation']}")
    
    if result['recommendations']:
        st.subheader("💡 Recommendations")
        for rec in result['recommendations']:
            st.markdown(f"- {rec}")
    
    st.divider()
    st.caption("Built with Streamlit | Models: GPT-2, BART-large-MNLI | Fact Source: Wikipedia API")

elif analyze_button and not text_input:
    st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("*Note: This tool provides indicators, not definitive truth. Always verify important claims.*")
