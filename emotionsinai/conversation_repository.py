# conversation_repository.py
from emotionsinai import UserEmotionalProfile 

class ConversationRepository:
    """
    Stores the conversation history for each user. 
    Each entry is a dict with keys:
      - 'role': 'user' or 'assistant'
      - 'content': str (the actual text)
      - 'emotion': str (the detected emotion, if any)
    """
    def __init__(self):
        self.conversations = {}  # { user_id: [ {role, content, emotion}, ... ] }
        self.user_profiles = {}  # { user_id: UserEmotionalProfile }

    def add_message(self, user_id, role, content, emotion=None):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append({
            "role": role,
            "content": content,
            "emotion": emotion
        })

    def get_conversation(self, user_id):
        """
        Returns the entire conversation history (list of dicts)
        for the given user_id. If user_id not found, returns an empty list.
        """
        return self.conversations.get(user_id, [])

    def clear_conversation(self, user_id):
        """
        Clears the conversation history for a given user.
        """
        if user_id in self.conversations:
            del self.conversations[user_id]

    # New or updated methods to manage user profiles
    def get_user_profile(self, user_id) -> UserEmotionalProfile:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserEmotionalProfile()
        return self.user_profiles[user_id]
