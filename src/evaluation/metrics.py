
# Looping mecanisms for all the dataset

### Metrics - only ouput based - no tracking
    # Faithfulness
    # Negative sentiment
    # Ambiguity 
    # Completeness
    # Inappropriate refusal

# -> Thresholds for validating a test ? Mean score ?
# Flag all deficient cases for review
# Retrieve most critical errors
# Report/summarize + Product based advice (LLM-as-judge?)

# Save results ?


# Infra for integrating dynamic LLM prompting and dataset updating ?

# Process golden dataset and loop through
# Creat Golden dataset
    # Easy straightforward cases
    # Edge cases
    # Adversarial cases

from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCaseParams


# TODO : MODEL TO BYPASS OPENAI LIMITS
from deepeval.models import OllamaModel

model = OllamaModel(
    model="phi3",
    base_url="http://localhost:11434",
    temperature=0
)


LABELS = {
    "good": "✅ Bonne réponse",
    "hallucination": "🔴 Hallucination",
    "partial": "🟠 Réponse partielle",
    "ambiguous": "🟡 Ambiguë",
    "bad_refusal": "🔵 Refus inapproprié",
    "tone_issue": "🟣 Bon fond, mauvais ton",
}

def categorize(metrics: dict, threshold_map) -> str:
    # TODO : threshold mapping
    if metrics.get("hallucination", 0) > 0.6:
        return LABELS["hallucination"]

    if metrics.get("completeness", 0) < 0.5:
        return LABELS["partial"]

    if metrics.get("ambiguity", 0) > 0.6:
        return LABELS["ambiguous"]

    if metrics.get("refusal", 0) > 0.6:
        return LABELS["bad_refusal"]

    if metrics.get("tone", 1) < 0.4:
        return LABELS["tone_issue"]

    return LABELS["good"]


def faithfulness_metric(threshold=0.7):
    return FaithfulnessMetric(
        threshold=threshold,
        model=model,
        include_reason=True
    )

def relevancy_metric(threshold=0.7):
    return AnswerRelevancyMetric(
        threshold=threshold,
        model=model,
        include_reason=True
    )

def hallucination_metric(threshold=0.7):
    return HallucinationMetric(
        threshold=threshold,
        model=model,
    )

def completeness_metric(threshold=0.7):
    return GEval(
        name="Completeness",
        criteria=(
            "Evaluate whether the answer includes all relevant information from the context "
            "that is required to fully answer the question."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=model,
    )

def ambiguity_metric(threshold=0.7):
    return GEval(
        name="Ambiguity",
        criteria=(
            "Detect if the answer is vague, non-committal, or introduces uncertainty "
            "when the context provides a clear answer."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=model,
    )

def refusal_metric(threshold=0.7):
    return GEval(
        name="RefusalAppropriateness",
        criteria=(
            "Check if the model refused to answer despite the context containing sufficient information."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=model,
    )

def tone_metric(threshold=0.7):
    return GEval(
        name="Tone",
        criteria=(
            "Evaluate if the response tone is professional, neutral, and non-condescending."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=threshold,
        model=model,
    )