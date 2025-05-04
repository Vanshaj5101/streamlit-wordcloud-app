import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from utils.file_handler import load_csv
from utils.gpt_helper import get_gpt_response
from utils.text_cleaner import preprocess_text


class FeedbackAnalyzerApp:
    def __init__(self):
        self.default_prompt = self.load_default_prompt()
        self.df = None
        self.prompt = ""
        self.uploaded_file = None

    def load_default_prompt(self):
        with open("prompts/gpt_sentiment_prompt.txt", "r") as file:
            return file.read()

    def init_session_state(self):
        if "prompt_text" not in st.session_state:
            st.session_state.prompt_text = self.default_prompt

    def render_file_upload(self):
        self.uploaded_file = st.file_uploader(
            "📁 Upload your CSV file", type=["csv"], accept_multiple_files=False
        )

    def render_prompt_input(self):
        self.prompt = st.text_area(
            "✍️ Enter your GPT prompt",
            value=st.session_state.prompt_text,
            key="prompt_text",
            height=200,
        )

    def render_file_preview(self):
        st.subheader("📄 Preview of Uploaded Feedback")
        st.dataframe(self.df.head())

    def render_wordcloud_section(self):
        st.subheader("☁️ Word Cloud from Feedback")
        try:
            cleaned_text = preprocess_text(self.df["comment"])
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="viridis",
                max_words=100,
            ).generate(cleaned_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating word cloud: {e}")

    def render_gpt_analysis_section(self):
        st.subheader("📝 Key Takeaways from GPT Analysis")
        status_box = st.empty()
        try:
            with st.spinner("Initializing analysis..."):
                gpt_response = get_gpt_response(system_prompt=self.prompt, df=self.df)
            status_box.info("🔍 Summarizing qualitative themes...")
            time.sleep(1.2)
            status_box.info("📊 Organizing feedback by sentiment...")
            time.sleep(1.2)
            status_box.info("🧠 Finalizing summary report...")
            time.sleep(1.2)
            status_box.success("✅ Insight report ready!")
            st.markdown(gpt_response)
        except Exception as e:
            st.error(f"Error during GPT analysis: {e}")

    def run(self):
        st.set_page_config(page_title="Qualitative Feedback Insight Generator", layout="centered")
        st.title("📈 Qualitative Feedback Insight Generator")
        st.markdown("Upload a CSV of qualitative comments and generate a GPT-powered summary and visual word cloud for your presentation.")

        self.init_session_state()
        self.render_file_upload()
        self.render_prompt_input()

        col1, col2 = st.columns([1, 1])
        with col1:
            analyze_clicked = st.button("🔍 Analyze", use_container_width=True)
        with col2:
            reset_clicked = st.button(
                "🔄 Reset to Default Prompt", use_container_width=True
            )
            if reset_clicked:
                st.session_state.prompt_text = self.default_prompt

        if analyze_clicked:
            if not self.uploaded_file or not self.prompt.strip():
                st.warning("Please upload a CSV file and enter a prompt.")
                return

            try:
                self.df = load_csv(self.uploaded_file)
                st.success("File uploaded successfully.")
                self.render_file_preview()
                self.render_wordcloud_section()
                self.render_gpt_analysis_section()

            except Exception as e:
                st.error(f"Error loading or processing file: {e}")
