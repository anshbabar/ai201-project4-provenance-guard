import json
import math
import os
import re
import statistics
from typing import Any

from groq import Groq


AI_TRANSITION_PHRASES = [
    "furthermore",
    "moreover",
    "additionally",
    "in conclusion",
    "it is important to note",
    "it is essential to",
    "overall",
    "in today's world",
    "plays a crucial role",
    "transformative paradigm",
    "stakeholders",
]


HIGH_CONFIDENCE_AI_LABEL = (
    "This content is likely AI-generated. Automated analysis found strong "
    "AI-like patterns in the writing. This result is an estimate, not proof "
    "of authorship, and the creator may appeal the classification."
)

HIGH_CONFIDENCE_HUMAN_LABEL = (
    "This content is likely human-written. Automated analysis found strong "
    "indicators of human authorship, but no automated detector can verify "
    "authorship with certainty."
)

UNCERTAIN_LABEL = (
    "The authorship of this content is uncertain. Automated analysis found "
    "mixed or inconclusive signals, so no confident attribution has been made."
)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Restrict a number to the provided range.
    """
    return max(minimum, min(value, maximum))


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a model response.

    This supports responses containing markdown code blocks or extra text.
    """
    cleaned = raw_text.strip()

    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            raise ValueError("The model response did not contain valid JSON.")

        return json.loads(match.group())


def analyze_with_groq(text: str) -> dict:
    """
    Use Groq to estimate the likelihood that text was AI-generated.

    Returns:
        {
            "score": float from 0.0 to 1.0,
            "reasoning": str,
            "available": bool
        }

    If the API is unavailable, a neutral score of 0.5 is returned.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "score": 0.5,
            "reasoning": "Groq API key was unavailable, so a neutral score was used.",
            "available": False,
        }

    prompt = f"""
You are one signal in a content-attribution system.

Analyze the writing below and estimate how likely it is to have been generated
primarily by an AI system.

Important rules:
- Do not claim certainty.
- Formal writing is not automatically AI-generated.
- Personal, informal writing is not automatically human-written.
- Consider formulaic transitions, generic phrasing, excessive uniformity,
  lack of concrete personal detail, sentence consistency, and overall style.
- Return only valid JSON.
- The score must be between 0.0 and 1.0.
- 0.0 means strongly human-like.
- 0.5 means uncertain.
- 1.0 means strongly AI-like.

Return exactly this structure:
{{
  "score": 0.0,
  "reasoning": "One or two brief sentences."
}}

Text:
{text}
"""

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate writing patterns cautiously and return "
                        "structured JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=250,
        )

        raw_response = response.choices[0].message.content or ""

        parsed = extract_json_object(raw_response)

        score = clamp(float(parsed.get("score", 0.5)))
        reasoning = str(
            parsed.get(
                "reasoning",
                "The model did not provide an explanation.",
            )
        )

        return {
            "score": round(score, 4),
            "reasoning": reasoning,
            "available": True,
        }

    except Exception as error:
        return {
            "score": 0.5,
            "reasoning": (
                "The Groq signal was unavailable, so a neutral score was used. "
                f"Error: {type(error).__name__}"
            ),
            "available": False,
        }


def split_sentences(text: str) -> list[str]:
    """
    Split text into non-empty sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> list[str]:
    """
    Extract lowercase word tokens.
    """
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def normalize_metric(
    value: float,
    low: float,
    high: float,
    reverse: bool = False,
) -> float:
    """
    Convert a metric into a score from 0.0 to 1.0.

    Values at or below low become 0.0.
    Values at or above high become 1.0.
    """
    if high <= low:
        return 0.5

    normalized = clamp((value - low) / (high - low))

    if reverse:
        normalized = 1.0 - normalized

    return normalized


def analyze_stylometrics(text: str) -> dict:
    """
    Analyze structural writing features and return an AI-likelihood score.
    """
    words = tokenize_words(text)
    sentences = split_sentences(text)

    word_count = len(words)
    sentence_count = len(sentences)

    if word_count == 0:
        return {
            "score": 0.5,
            "metrics": {
                "word_count": 0,
                "sentence_count": 0,
                "average_sentence_length": 0.0,
                "sentence_length_variance": 0.0,
                "type_token_ratio": 0.0,
                "punctuation_density": 0.0,
                "transition_phrase_count": 0,
            },
            "note": "No analyzable words were found.",
        }

    sentence_lengths = [
        len(tokenize_words(sentence))
        for sentence in sentences
        if tokenize_words(sentence)
    ]

    if not sentence_lengths:
        sentence_lengths = [word_count]

    average_sentence_length = statistics.mean(sentence_lengths)

    if len(sentence_lengths) > 1:
        sentence_length_variance = statistics.pvariance(sentence_lengths)
    else:
        sentence_length_variance = 0.0

    unique_word_count = len(set(words))
    type_token_ratio = unique_word_count / word_count

    punctuation_count = len(re.findall(r"[.,!?;:—\-()]", text))
    punctuation_density = punctuation_count / word_count

    lowered_text = text.lower()

    transition_phrase_count = sum(
        lowered_text.count(phrase)
        for phrase in AI_TRANSITION_PHRASES
    )

    # Lower sentence-length variance can suggest overly uniform writing.
    uniformity_score = normalize_metric(
        sentence_length_variance,
        low=5.0,
        high=80.0,
        reverse=True,
    )

    # Low vocabulary diversity can suggest repetitive writing.
    vocabulary_score = normalize_metric(
        type_token_ratio,
        low=0.35,
        high=0.80,
        reverse=True,
    )

    # Moderate punctuation is common. Extremely low punctuation may indicate
    # generated or list-like writing, though this signal receives low weight.
    punctuation_score = normalize_metric(
        punctuation_density,
        low=0.015,
        high=0.12,
        reverse=True,
    )

    transition_score = clamp(transition_phrase_count / 4.0)

    # Longer, highly regular sentences may increase the AI-like score.
    sentence_length_score = normalize_metric(
        average_sentence_length,
        low=8.0,
        high=28.0,
    )

    stylometric_score = (
        uniformity_score * 0.30
        + vocabulary_score * 0.25
        + punctuation_score * 0.10
        + transition_score * 0.20
        + sentence_length_score * 0.15
    )

    # Short text does not provide enough reliable stylometric evidence.
    # Pull the score toward uncertainty.
    if word_count < 25:
        short_text_weight = word_count / 25
        stylometric_score = (
            stylometric_score * short_text_weight
            + 0.5 * (1 - short_text_weight)
        )

    return {
        "score": round(clamp(stylometric_score), 4),
        "metrics": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_sentence_length": round(average_sentence_length, 4),
            "sentence_length_variance": round(
                sentence_length_variance,
                4,
            ),
            "type_token_ratio": round(type_token_ratio, 4),
            "punctuation_density": round(punctuation_density, 4),
            "transition_phrase_count": transition_phrase_count,
        },
        "note": (
            "Stylometric scores estimate structural similarity and do not "
            "prove authorship."
        ),
    }


def combine_scores(
    llm_score: float,
    stylometric_score: float,
    word_count: int,
    llm_available: bool,
) -> float:
    """
    Combine the two independent AI-likelihood signals.

    The planned weighting is:
        70% Groq LLM
        30% stylometric analysis

    When Groq is unavailable, the final score is pulled toward 0.5 to avoid
    making a confident decision from only one signal.
    """
    combined = llm_score * 0.70 + stylometric_score * 0.30

    if not llm_available:
        combined = combined * 0.45 + 0.5 * 0.55

    # Very short text should remain close to uncertain.
    if word_count < 15:
        evidence_weight = word_count / 15
        combined = combined * evidence_weight + 0.5 * (1 - evidence_weight)

    return round(clamp(combined), 4)


def classify_score(ai_likelihood: float) -> str:
    """
    Convert the AI-likelihood score into one of three attribution categories.
    """
    if ai_likelihood >= 0.70:
        return "likely_ai"

    if ai_likelihood <= 0.30:
        return "likely_human"

    return "uncertain"


def calculate_confidence(
    ai_likelihood: float,
    attribution: str,
) -> float:
    """
    Calculate confidence in the selected attribution category.

    This is separate from AI likelihood:
    - AI likelihood describes the direction of the result.
    - Confidence describes the strength of the classification.
    """
    if attribution == "likely_ai":
        confidence = 0.5 + ((ai_likelihood - 0.70) / 0.30) * 0.5

    elif attribution == "likely_human":
        confidence = 0.5 + ((0.30 - ai_likelihood) / 0.30) * 0.5

    else:
        distance_from_center = abs(ai_likelihood - 0.5)
        confidence = 0.5 - distance_from_center

    return round(clamp(confidence), 4)


def get_transparency_label(attribution: str) -> str:
    """
    Return the exact required transparency-label text.
    """
    if attribution == "likely_ai":
        return HIGH_CONFIDENCE_AI_LABEL

    if attribution == "likely_human":
        return HIGH_CONFIDENCE_HUMAN_LABEL

    return UNCERTAIN_LABEL


def analyze_text(text: str) -> dict:
    """
    Run the full multi-signal detection pipeline.
    """
    llm_result = analyze_with_groq(text)
    stylometric_result = analyze_stylometrics(text)

    word_count = stylometric_result["metrics"]["word_count"]

    ai_likelihood = combine_scores(
        llm_score=llm_result["score"],
        stylometric_score=stylometric_result["score"],
        word_count=word_count,
        llm_available=llm_result["available"],
    )

    attribution = classify_score(ai_likelihood)

    confidence = calculate_confidence(
        ai_likelihood=ai_likelihood,
        attribution=attribution,
    )

    label = get_transparency_label(attribution)

    return {
        "attribution": attribution,
        "confidence": confidence,
        "ai_likelihood": ai_likelihood,
        "label": label,
        "signals": {
            "llm_score": llm_result["score"],
            "llm_reasoning": llm_result["reasoning"],
            "llm_available": llm_result["available"],
            "stylometric_score": stylometric_result["score"],
            "stylometric_metrics": stylometric_result["metrics"],
        },
    }