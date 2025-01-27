import math
from typing import Dict, List

class UserEmotionalProfile:
    """
    Stores aggregated emotion scores for a user over time.
    Example schema for each message's emotion:
      {
        "trust": 0.65,
        "frustration": 0.20,
        "positivity": 0.70,
        ...
      }
    We maintain:
      - rolling_averages: current averages of each emotion
      - message_history: list of past emotion dictionaries
    """

    def __init__(self):
        self.rolling_averages: Dict[str, float] = {}
        self.message_history: List[Dict[str, float]] = []

    def update_emotions(self, new_emotions: Dict[str, float]):
        """
        Incorporates new emotion scores into rolling averages 
        and appends to message_history.
        """
        self.message_history.append(new_emotions)

        for emotion_key, value in new_emotions.items():
            old_avg = self.rolling_averages.get(emotion_key, None)
            if old_avg is None:
                # first time setting average
                self.rolling_averages[emotion_key] = value
            else:
                # simple incremental average: avg' = (old_avg * n + value) / (n+1)
                n = len(self.message_history) - 1  # n prior messages
                new_avg = (old_avg * n + value) / (n + 1)
                self.rolling_averages[emotion_key] = new_avg

    def detect_outliers(self, new_emotions: Dict[str, float], threshold: float = 0.3) -> List[str]:
        """
        Compare new emotion values to the rolling averages.
        If difference is above 'threshold', we flag it as an outlier.
        Return a list of emotion keys that deviate significantly.
        """
        outlier_keys = []
        for emotion_key, value in new_emotions.items():
            avg_val = self.rolling_averages.get(emotion_key, 0.5)  # default to 0.5 if not found
            if abs(value - avg_val) > threshold:
                outlier_keys.append(emotion_key)
        return outlier_keys
