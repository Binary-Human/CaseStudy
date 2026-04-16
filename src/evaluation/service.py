"""Service to process metrics, called by dashboard"""

from deepeval.test_case import LLMTestCase
from deepeval import assert_test

import pandas as pd
from evaluation.metrics import (faithfulness_metric,relevancy_metric,hallucination_metric,refusal_metric,tone_metric,categorize)

import streamlit as st

# Used for caching
@st.cache_data(show_spinner=False)
def _evaluate_case_cached(row_dict, context_df):
    context = retrieve_context(row_dict["question"], context_df)

    test_case = LLMTestCase(
        input=row_dict["question"],
        actual_output=row_dict["answer"],
        retrieval_context=[c["detail"] for c in context],
        context=[c["detail"] for c in context]
    )

    return {
        "id": row_dict["id"],
        "relevancy": relevancy_metric().measure(test_case),
        "faithfulness": faithfulness_metric().measure(test_case),
        "hallucination": hallucination_metric().measure(test_case),
        "refusal": refusal_metric().measure(test_case),
        "tone": tone_metric().measure(test_case),
        "context": context,
    }

class EvaluationService:

    def __init__(self, context_df):
        self.context_df = context_df

    def evaluate_case(self, row):
        print(f"Evaluating case {row['id']}")

        return _evaluate_case_cached(
            row.to_dict(),
            self.context_df,
        )

def retrieve_context(question: str, context_df: pd.DataFrame, top_k: int = 5):
    """
    Key word based context retrieval
    """
    scores = []

    # Parses words in the questions and looks for any match in context base

    for _, row in context_df.iterrows():
        text = " ".join(str(v) for v in row.values).lower()
        score = sum(word in text for word in question.lower().split())
        scores.append(score)

    context_df = context_df.copy()
    context_df["score"] = scores

    return context_df.sort_values("score", ascending=False).head(top_k).to_dict("records")