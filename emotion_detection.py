import requests  # Import the requests library to handle HTTP requests
import json

def emotion_detector(text_to_analyse: str) -> dict:
    """
    Sends text to Watson NLP emotion predict API and returns the result.

    Args:
        text_to_analyse (str): The text to analyze.

    Returns:
        dict: A dictionary with keys 'label' and 'score'.
    """
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    payload = {"raw_document": {"text": text_to_analyse}}
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        data = json.loads(response.text)
        emotions = data["emotionPredictions"][0]["emotion"]
        anger_score = emotions["anger"]
        disgust_score = emotions["disgust"]
        fear_score = emotions["fear"]
        joy_score = emotions["joy"]
        sadness_score = emotions["sadness"]
        dominant_emotion = max(emotions, key=emotions.get)
        return {
            'anger': anger_score,
            'disgust': disgust_score,
            'fear': fear_score,
            'joy': joy_score,
            'sadness': sadness_score,
            'dominant_emotion': dominant_emotion
        }
    except requests.exceptions.RequestException:
        return None
