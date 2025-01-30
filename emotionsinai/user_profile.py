from typing import Dict, List, Optional

class UserProfile:
    """
    A unified user profile that stores both:
      - The user's overall emotional profile with rolling averages.
      - The user's conversation history with emotion metadata.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.rolling_averages: Dict[str, float] = {}
        self.message_history: List[Dict[str, float]] = []
        self.conversations: List[Dict[str, Optional[str]]] = []

    def add_message(self, role: str, content: str, emotions: Optional[Dict[str, float]] = None):
        """
        Adds a new message to the conversation history along with its emotions.
        Updates the user's emotional profile based on the new emotions.
        """
        message_entry = {
            "role": role,
            "content": content,
            "emotions": emotions
        }
        self.conversations.append(message_entry)

        if emotions:
            self.update_emotions(emotions)

    def get_conversation_history(self, num_messages: Optional[int] = None) -> List[Dict[str, Optional[str]]]:
        """
        Returns the conversation history. 
        If num_messages is provided, return only the last 'num_messages' messages.
        """
        if num_messages is None or num_messages >= len(self.conversations):
            return self.conversations
        return self.conversations[-num_messages:]

    def clear_conversation_history(self):
        """
        Clears the conversation history.
        """
        self.conversations = []

    def update_emotions(self, new_emotions: Dict[str, float]):
        """
        Incorporates new emotion scores into rolling averages 
        and appends to message history.
        """
        self.message_history.append(new_emotions)

        for emotion_key, value in new_emotions.items():
            old_avg = self.rolling_averages.get(emotion_key, None)
            if old_avg is None:
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

    def get_emotional_profile(self) -> Dict[str, float]:
        """
        Returns the user's current emotional profile (rolling average emotions).
        """
        return self.rolling_averages.copy()
