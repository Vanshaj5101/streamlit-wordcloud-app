import streamlit as st
import pandas as pd

from utils.file_handler import load_csv
from utils.gpt_helper import get_gpt_response

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils.text_cleaner import preprocess_text

import time

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

# --- Side-by-side Buttons: Analyze & Reset ---
# --- Side-by-side Buttons: Analyze & Reset ---
col1, col2 = st.columns([1, 1])

with col1:
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True)

with col2:
    reset_clicked = st.button("🔄 Reset to Default Prompt", use_container_width=True)
    if reset_clicked:
        st.session_state.prompt_text = DEFAULT_SENTIMENT_PROMPT

# --- Analyze Button ---
if analyze_clicked:
    if not uploaded_file or not prompt.strip():
        st.warning("Please upload a CSV file and enter a prompt.")
    else:
        try:
            df = load_csv(uploaded_file)
            st.success("File uploaded successfully.")
            st.subheader("📄 File Preview")
            st.dataframe(df.head())

            # Placeholder for Word Cloud
            st.subheader("☁️ Word Cloud")
            try:
                cleaned_text = preprocess_text(df["comment"])

                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(cleaned_text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating word cloud: {e}")

            # GPT Analysis
            st.subheader("🧠 GPT Analysis")
            status_box = st.empty()  # Placeholder for dynamic updates

            try:
                with st.spinner("Initializing analysis..."):
                    gpt_response = get_gpt_response(system_prompt=prompt, df=df)
                status_box.info("🔍 Analyzing comments...")
                time.sleep(1.2)

                status_box.info("📊 Extracting key insights...")
                time.sleep(1.2)

                status_box.info("🧠 Detecting sentiment and preparing summary...")
                time.sleep(1.2)


                status_box.success("✅ Insight report ready!")
                st.markdown(gpt_response)
            except Exception as e:
                st.error(f"Error during GPT analysis: {e}")

        except Exception as e:
            st.error(f"Error loading file: {e}")
