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

    def add_message(self, role: str, content: str, emotions: Optional[List[Dict[str, float]]] = None):
        """
        Adds a new message to the conversation history along with its emotions.
        Updates the user's emotional profile based on the new emotions.
        """
        message_entry = {
            "role": role,
            "content": content,
            "emotions": emotions
        }

        print(f"Adding message: {message_entry}")
        self.conversations.append(message_entry)

        if emotions:
            #self.detect_outliers(emotions)
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

    def update_emotions(self, new_emotions: List[Dict[str, float]]):
        """
        Incorporates new emotion scores into the agent's emotional representation
        using an exponential moving average (EMA) update rule, which is more 
        psychologically realistic than a simple arithmetic mean. The EMA update 
        accounts for emotional inertia: emotions such as trust or pride tend to be 
        more stable (low alpha), while emotions like anger or fear can be more volatile 
        (high alpha).
        
        Expects new_emotions to be a list of dictionaries with 'emotion' and 'score' keys.
        For example:
            [
                {"emotion": "happiness", "score": 0.8},
                {"emotion": "anger", "score": 0.3},
                ...
            ]
        
        The updated rolling averages are computed as:
            new_avg = (1 - alpha) * old_avg + alpha * new_score
            
        where alpha is a dynamic learning rate that may vary for each emotion.
        """
        # Define dynamic learning rates for different emotions based on their emotional inertia.
        # Lower alpha means slower update (more inertia), higher alpha means faster change.
        alpha_values = {
            "happiness": 0.2,
            "sadness": 0.3,
            "anger": 0.4,
            "fear": 0.4,
            "surprise": 0.3,
            "disgust": 0.3,
            "love": 0.2,
            "jealousy": 0.4,
            "guilt": 0.3,
            "pride": 0.2,
            "shame": 0.3,
            "compassion": 0.2,
            "sympathy": 0.2,
            "trust": 0.1
        }
        
        # For traceability, convert the list of emotion updates into a dictionary
        # and append it to the message history.
        current_update = {emotion_obj['emotion']: emotion_obj['score'] for emotion_obj in new_emotions}
        self.message_history.append(current_update)
        
        # Update each emotion using the exponential moving average approach.
        for emotion, new_score in current_update.items():
            # Retrieve the corresponding alpha value, defaulting to 0.3 if not defined.
            alpha = alpha_values.get(emotion, 0.3)
            # If this emotion has not been updated before, initialize it with the new score.
            if emotion not in self.rolling_averages:
                self.rolling_averages[emotion] = new_score
            else:
                # EMA update: new_average = (1 - alpha) * old_average + alpha * new_score
                old_avg = self.rolling_averages[emotion]
                new_avg = (1 - alpha) * old_avg + alpha * new_score
                self.rolling_averages[emotion] = new_avg

        # Optionally, you could log or return the updated rolling averages for further analysis.


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
