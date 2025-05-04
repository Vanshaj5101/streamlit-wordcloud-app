import openai
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_df_as_comment_block(df: pd.DataFrame) -> str:
    """
    Convert the DataFrame into a structured comment block for GPT.
    """
    rows = [
        f"Record ID: {row['record_id']}, Course: {row['course_id']}, Comment: {row['comment']}"
        for _, row in df.iterrows()
    ]
    return "\n".join(rows)


def get_gpt_response(system_prompt: str, df: pd.DataFrame) -> str:
    """
    Send structured feedback to OpenAI for qualitative analysis.
    """
    comment_data = format_df_as_comment_block(df)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Comments:\n{comment_data}"},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content
