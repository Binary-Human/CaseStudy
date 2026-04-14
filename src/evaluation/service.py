from deepeval.test_case import LLMTestCase
from deepeval import assert_test

import pandas as pd
from evaluation.metrics import (faithfulness_metric,relevancy_metric,hallucination_metric,completeness_metric,ambiguity_metric,refusal_metric,categorize)

class EvaluationService:

    def __init__(self, context_df):
        self.context_df = context_df

    def _run_metric(self, metric, test_case):
        metric.measure(test_case)
        return metric.score, metric.reason

    def evaluate_case(self, row):

        context = retrieve_context(row["question"], self.context_df)

        print("Context retrieved:", context)

        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=[c["detail"] for c in context]
        )

        # TODO :  test with others
        """ 
            "faithfulness": faithfulness_metric().measure(test_case), # TODO : need to add reason parameters? 
            "hallucination": hallucination_metric().measure(test_case),
            "completeness": completeness_metric().measure(test_case),
            "ambiguity": ambiguity_metric().measure(test_case),
            "refusal": refusal_metric().measure(test_case),
        """

        return {
            "id": row["id"],
            "relevancy": relevancy_metric().measure(test_case),
            "context": context
        }

    def run_dataset(self, df):
        results = []
        for _, row in df.iterrows():
            results.append(self.evaluate_case(row))
        return results

def retrieve_context(question: str, context_df: pd.DataFrame, top_k: int = 5):
    """
    Key word based context retrieval
    """
    scores = []

    for _, row in context_df.iterrows():
        text = " ".join(str(v) for v in row.values).lower()
        score = sum(word in text for word in question.lower().split())
        scores.append(score)

    context_df = context_df.copy()
    context_df["score"] = scores

    return context_df.sort_values("score", ascending=False).head(top_k).to_dict("records")