import streamlit as st
import pandas as pd

from utils.file_handler import load_csv
from utils.gpt_helper import get_gpt_response

# --- Page Setup ---
st.set_page_config(page_title="WordCloud & GPT Analyzer", layout="centered")
st.title("📊 WordCloud & GPT Text Analyzer")
st.markdown("Upload a CSV and enter a custom prompt to analyze your text data.")

# Load default system prompt
with open("prompts/gpt_sentiment_prompt.txt", "r") as file:
    DEFAULT_SENTIMENT_PROMPT = file.read()

# --- Initialize session state ---
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = DEFAULT_SENTIMENT_PROMPT

# --- Reset Prompt Button ---
if st.button("🔄 Reset to Default Prompt"):
    st.session_state.prompt_text = DEFAULT_SENTIMENT_PROMPT

# --- File Upload ---
uploaded_file = st.file_uploader(
    "📁 Upload your CSV file", type=["csv"], accept_multiple_files=False
)

# --- Prompt Input ---
prompt = st.text_area(
    "✍️ Enter your GPT prompt",
    value=st.session_state.prompt_text,
    key="prompt_text",
    height=200,
)

# --- Analyze Button ---
if st.button("🔍 Analyze"):
    if not uploaded_file or not prompt.strip():
        st.warning("Please upload a CSV file and enter a prompt.")
    else:
        try:
            df = load_csv(uploaded_file)
            st.success("File uploaded successfully.")
            st.subheader("📄 File Preview")
            st.dataframe(df.head())

            # GPT Analysis
            st.subheader("🧠 GPT Analysis")
            try:
                with st.spinner("Analyzing with GPT..."):
                    gpt_response = get_gpt_response(system_prompt=prompt, df=df)
                st.markdown(gpt_response)
            except Exception as e:
                st.error(f"Error during GPT analysis: {e}")

            # Placeholder for Word Cloud
            st.subheader("☁️ Word Cloud")
            st.info("Word cloud will be generated here in Phase 4.")

        except Exception as e:
            st.error(f"Error loading file: {e}")
