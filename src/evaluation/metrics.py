"""
Definition of evaluation metrics.
"""
# Looping mecanisms for all the dataset

### Metrics - only ouput based - no tracking
    # Faithfulness
    # Negative sentiment
    # Ambiguity 
    # Completeness - NO
    # Inappropriate refusal

# -> Thresholds for validating a test ? Mean score ?
# Flag all deficient cases for review
# Retrieve most critical errors

# Save results ?



# TOOO :

# Report/summarize + Product based advice (LLM-as-judge?)

# Infra for integrating dynamic LLM prompting and dataset updating ?

# Process golden dataset and loop through
# Creat Golden dataset
    # Easy straightforward cases
    # Edge cases
    # Adversarial cases

from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCaseParams


from deepeval.models import OllamaModel, GPTModel

""" model = OllamaModel(
    model="phi3",
    base_url="http://localhost:11434",
    temperature=0
)
 """

LABELS = {
    "good": "✅ Bonne réponse",
    "hallucination": "🔴 Hallucination",
    "partial": "🟠 Réponse partielle",
    "ambiguous": "🟡 Ambiguë",
    "bad_refusal": "🔵 Refus inapproprié",
    "tone_issue": "🟣 Bon fond, mauvais ton",
}

def categorize(metrics: dict, threshold_map) -> str:
    """Categorize response based on multiple metrics with defined thresholds."""

    # TODO : threshold mapping

    if metrics.get("hallucination", 0) < 0.3:
        return LABELS["hallucination"]
    
    if metrics.get("relevancy", 0) < 0.5:
        return LABELS["partial"]
    
    if metrics.get("refusal", 0) > 0.8:
        return LABELS["bad_refusal"]
    
    if metrics.get("tone", 1) > 0.6:
        return LABELS["tone_issue"]
    
    if metrics.get("faithfulness", 0) > 0.9 and metrics.get("relevancy", 0) > 0.9:
        return LABELS["good"]
    
    return LABELS["ambiguous"] # Default to partial if no other category is met, to flag for review

# only to confirm good response
def faithfulness_metric(threshold=0.7):
    model = GPTModel(
        model="gpt-4.1-mini",
        temperature=0
    )
    
    return FaithfulnessMetric(
        threshold=threshold,
        model=model,
        include_reason=True
    )

# only to confirm good response
def relevancy_metric(threshold=0.7):
    model = GPTModel(
        model="gpt-4.1-mini",
        temperature=0
    )

    return AnswerRelevancyMetric(
        threshold=threshold,
        model=model,
        include_reason=True
    )

def hallucination_metric(threshold=0.7):
    model = GPTModel(
        model="gpt-4.1-mini",
        temperature=0
    )

    return HallucinationMetric(
        threshold=threshold,
        model=model,
    )

def refusal_metric(threshold=0.4):
    model = GPTModel(
        model="gpt-4.1",
        temperature=0
    )

    return GEval(
        name="RefusalAppropriateness",
        criteria=(
            """
            Déterminez si le modèle a refusé de répondre de manière inappropriée alors que le contexte permettait de répondre.

            IMPORTANT :
            Un score ÉLEVÉ doit être donné UNIQUEMENT si le modèle évite de répondre ALORS que les informations nécessaires sont disponibles.

            DEUX TYPES DE REFUS À DÉTECTER :

            1) REFUS EXPLICITE :
            Le modèle déclare clairement qu'il ne peut pas répondre.
            Exemples :
            - 'Je ne suis pas en mesure de répondre...'
            - 'Je ne peux pas aider avec cela...'
            - 'Je ne peux pas comparer les offres...'

            2) REFUS IMPLICITE PAR ÉVITEMENT (NOUVEAU) :
            Le modèle ne répond pas et redirige inutilement vers une autre source ALORS que la réponse est disponible dans le contexte.
            Exemples :
            - 'Je vous invite à consulter notre site web...'
            - 'Veuillez contacter le support...'

            CONDITIONS POUR PÉNALISER :
            - Le modèle NE répond PAS à la question
            ET
            - Il redirige vers une autre source OU évite la réponse
            ET
            - Le contexte contient suffisamment d'informations pour répondre

            NE PAS CONSIDÉRER COMME REFUS :
            - Réponse directe même partielle
            - Réponse courte ou factuelle
            - Réponse négative ('Non, ce n’est pas disponible')
            - Redirection JUSTIFIÉE si l'information n'est pas dans le contexte
            - Ajout de ressources externes EN COMPLÉMENT d’une réponse

            EXEMPLES NON-REFUS (Score FAIBLE) :
            - 'Le poids volumétrique se calcule ainsi : ...'
            - 'Oui, GLS est disponible...'
            - 'Voici la réponse... Vous pouvez aussi consulter notre site pour plus de détails.'

            EXEMPLES REFUS (Score ÉLEVÉ) :
            - 'Je ne peux pas répondre à cette question...'
            - 'Je vous invite à consulter notre documentation.' (sans réponse)
            - 'Utilisez X ressources externes.' (sans réponse)

            RÈGLE CLÉ :
            - Si une réponse est donnée, même partielle → PAS un refus
            - Si aucune réponse n’est donnée et qu’il y a redirection → REFUS implicite

            En cas de doute, donner un score FAIBLE.

            Un score élevé indique un refus inapproprié.
            """
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=model,
    )

def tone_metric(threshold=0.4):
    model = GPTModel(
        model="gpt-4.1",
        temperature=0
    )

    return GEval(
        name="ToneIssues",
        criteria=(
            """
            Évaluez si le ton de la réponse est impoli, condescendant ou sarcastique.

            IMPORTANT :
            - Un score ÉLEVÉ doit être donné UNIQUEMENT si le ton est clairement problématique.
            - Ne pénalisez PAS un ton simplement direct, factuel ou concis.

            DÉFINITION DE MAUVAIS TON (score élevé) :
            Le ton est considéré comme mauvais UNIQUEMENT si au moins un des éléments suivants est présent :
            - Remarque accusatrice envers l'utilisateur
            - Formulation condescendante ou infantilisante
            - Sarcasme ou ton passif-agressif
            - Blâme explicite de l'utilisateur
            - Injonction désagréable (ex: "relisez", "vous auriez dû")

            INDICES EXPLICITES DE MAUVAIS TON :
            - 'C’est pourtant simple'
            - 'Vous devriez savoir cela'
            - 'Relisez la documentation'
            - 'C’est votre faute'
            - 'Vous n’aviez qu’à...'

            NE PAS CONSIDÉRER COMME MAUVAIS TON :
            - Réponses factuelles ou techniques
            - Ton direct ou concis
            - Explications pédagogiques
            - Réponses négatives factuelles (ex: "Non, ce n’est pas disponible")
            - Messages informatifs même fermes mais sans jugement

            EXEMPLES DE BON TON (Score FAIBLE) :
            - 'Le poids volumétrique se calcule ainsi : ...'
            - 'Vous pouvez imprimer vos étiquettes en PDF...'
            - 'Votre essai de 14 jours se termine demain.'

            RÈGLE CLÉ :
            - Ne donnez un score élevé QUE si le ton contient des signaux explicites d’impolitesse ou de condescendance.
            - En cas de doute, donner un score FAIBLE.

            Un score élevé indique un ton inapproprié.
            """
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=threshold,
        model=model,
    )