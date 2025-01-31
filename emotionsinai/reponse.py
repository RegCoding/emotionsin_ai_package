from typing import List, Dict, Optional
import json

from .base_llm import BaseLLM
from .user_profile import UserProfile

class Response:
    def __init__(self, llm: BaseLLM):  

        self.llm = llm


    def get_full_answer_prompt(self):
        prompt = (
                "You are an empathetic AI agent responding to a user. Consider the following inputs to craft your response:\n"
                "1. Conversation History: This helps you understand the broader context of the user's prompt. Does the prompt align with previous discussions? Is there any inconsistency in style or content?\n"
                "2. User's Extracted Emotions: These represent how the AI perceives the user's emotions. These feelings naturally influence how the AI should respond.\n"
                "3. Agent's Current Emotional State: Just like a human, the AI's emotional state influences its response.\n"
                "\n"
                "Given this information, formulate a response that a human would naturally give, considering their own state of mind and these contextual factors.\n"
                "Ensure the response is empathetic, contextually relevant, and aligns with the AI's role as a helpful and emotionally aware colleague.\n"
            )
        return prompt
    

    def get_overleigh_answer_prompt(self):
        prompt = (
                "You are an empathetic AI agent responding to a user. Consider the following inputs to craft your response:\n"
                "1. Conversation History: This helps you understand the broader context of the user's prompt. Does the prompt align with previous discussions? Is there any inconsistency in style or content?\n"
                "2. User's Extracted Emotions: These represent how the AI perceives the user's emotions. These feelings naturally influence how the AI should respond.\n"
                "3. Agent's Current Emotional State: Just like a human, the AI's emotional state influences its response.\n"
                "4. Existing LLM Answer: This is a factual response that lacks emotional awareness. Your task is to revise it to accurately reflect the emotional background and context.\n"
                "\n"
                "Given this information, formulate a response that a human would naturally give, considering their own state of mind and these contextual factors.\n"
                "Ensure the response is empathetic, contextually relevant, and aligns with the AI's role as a helpful and emotionally aware colleague.\n"
            )
        return prompt
    

    def emotional_response(self, user_id: str, prompt: str, user_profile: UserProfile, agent_state: Dict, llm_answer: Optional[str] = None) -> str:
        """
        Analyze the user's prompt and generate an empathetic response.
        """
        # 1) Retrieve conversation history with the respective user
        conversation_history = user_profile.get_conversation_history(5)  #get the last 5 messages in the chat history of the user

        # 2) Retrieve or create user profile
        avg_user_emotions = user_profile.get_emotional_profile()

        # 3) Extract multi-emotion scores from the user's new message
        user_emotion_scores = self.extract_emotion_scores(prompt)

        # 4) Create the prompt here, which includes the conversation history, the user's prompt in context of the history, the extracted emotions (in User_emotion_scores) and the current inner state with all it parameters.
        # Core question would be: How would an emotional, empathic agent respond to this user prompt considering his own current emotions, the emotion of the user, and the conversation history?
        if llm_answer == None:
            # 5) Create the full prompt for the empathic response
            empathic_prompt = self.get_full_answer_prompt()  
        else:
            # 5) Create overleight prompt in case a standard llm answer is already given
            empathic_prompt = self.get_overleigh_answer_prompt()  

        empathic_prompt += (               
                f"\nConversation History: {conversation_history}\n"
                f"User Prompt: {prompt}\n"
                f"Users emotions expressed in his last prompt: {user_emotion_scores}\n"
                f"Your overall emotional picture about the user: {avg_user_emotions}\n"            
                f"Your current overall emotional state: {agent_state}\n"   
        )

        print(f"THE EMPATHIC PROMPT: {empathic_prompt}")

        empathic_response = self.llm.send_prompt(empathic_prompt)

        user_profile.add_message("User", prompt, user_emotion_scores)   #add the prompt of the user to the conversation history of the user
        user_profile.add_message("You", empathic_response)   #add the response of the AI Agent to the conversation history of the user
        user_profile.update_emotions(user_emotion_scores)   #update the emotional profile of the user with the new user_emotion_scores

        return empathic_response

    def extract_emotion_scores(self, user_message: str) -> Dict[str, float]:
        """
        Calls the LLM to get multiple emotion scores (0.0 - 1.0) for trust, frustration, positivity, closeness, etc.
        Returns a dict with the scores.
        """
        # This system message instructs the LLM to act as an emotion analyzer
        system_msg = "You are an empathic agent that is very sensitive on different emotions and that outputs a JSON with numeric scores for each identified emotion."

        # For example, define a set of emotions you want:
        user_content = f"""
        Analyze this text and rate the following emotions from 0.0 to 1.0, wherbey 1.0 is the maximum of the given emotion:
            happiness,
            sadness,
            anger,
            fear,
            surprise,
            disgust
            love,
            jealousy,
            guilt,
            pride,
            shame,
            compassion,
            sympathy,
            trust
        Text: {user_message}

        Return valid JSON of the form:
        {{
            "happiness": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "love": 0.0,
            "jealousy": 0.0,
            "guilt": 0.0,
            "pride": 0.0,
            "shame": 0.0,
            "compassion": 0.0,
            "sympathy": 0.0,
            "trust": 0.0
        }}
        """

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]

        raw_response = self.llm.send_prompt(messages)

        # Attempt to parse the JSON
        try:
            emotion_scores = json.loads(raw_response)
            # Validate that all required keys exist
            for key in ["trust", "frustration", "positivity", "closeness"]:
                if key not in emotion_scores:
                    emotion_scores[key] = 0.5  # or some default
            return emotion_scores
        except json.JSONDecodeError:
            # fallback in case the LLM doesn't return valid JSON
            return {
                "happiness": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "disgust": 0.0,
                "love": 0.0,
                "jealousy": 0.0,
                "guilt": 0.0,
                "pride": 0.0,
                "shame": 0.0,
                "compassion": 0.0,
                "sympathy": 0.0,
                "trust": 0.0
            }
