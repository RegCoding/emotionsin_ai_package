from typing import List, Dict, Optional
import json

from .base_llm import BaseLLM
from .user_profile import UserProfile

class Response:
    def __init__(self, llm: BaseLLM):  
        self.llm = llm

    def get_combined_emotional_prompt(
        self, 
        conversation_history: str, 
        user_prompt: str, 
        avg_user_emotions: Dict, 
        agent_state: Dict, 
        llm_answer: Optional[str] = None
    ) -> str:
        """
        Create a prompt that instructs the LLM to both extract emotions and generate an adapted emotional answer.
        The prompt includes detailed instructions and expects a JSON output that contains ONLY the following three keys:
          - 'emotional_response': The final empathetic response as plain text.
          - 'reasoning': A detailed plain text explanation of how the answer was derived.
          - 'extracted_emotions': An array of objects, where each object contains:
              'emotion': the emotion name, and 
              'score': a numeric value between 0.0 and 1.0.
        """
        if llm_answer is None:
            base_prompt = (
                "You are an empathetic AI agent responding to a user. Consider the following inputs:\n"
                "1. User Prompt: The latest message from the user.\n"
                "2. Average User Emotions: Overall emotional profile derived from past interactions.\n"
                "3. Your own current Emotional State: Your own emotional state at the moment - Yes, you have emotions and can express it.\n"
                "\n"
                "Your tasks are:\n"
                "a) Analyze the user's prompt and extract all relevant emotions from the following list: "
                "happiness, sadness, anger, fear, surprise, disgust, love, jealousy, guilt, pride, shame, compassion, sympathy, trust. "
                "For each emotion, assign a score between 0.0 (absent) and 1.0 (very strong).\n"
                "b) Using the extracted emotions and the provided context, generate a final empathetic and emotionally adapted response.\n"
                "c) Provide a detailed explanation of your reasoning for integrating these emotional cues into your response.\n"
                "\n"
                "IMPORTANT: Return ONLY a valid JSON object with exactly these three keys:\n"
                "   'emotional_response' (plain text),\n"
                "   'reasoning' (plain text),\n"
                "   'extracted_emotions' (an array of objects where each object has 'emotion' and 'score', whereby 'score' can only be a value between 0.0 and 1.0).\n"
            )
        else:
            base_prompt = (
                "You are an empathetic AI agent tasked with refining an existing answer. Consider the following inputs:\n"
                "1. User Prompt: The latest message from the user.\n"
                "2. Average User Emotions: Overall emotional profile derived from past interactions.\n"
                "3. Agent's Current Emotional State: Your current emotional state.\n"
                "4. Existing LLM Answer: A factual response that currently lacks emotional depth.\n"
                "\n"
                "Your tasks are:\n"
                "a) Analyze the user's prompt and extract all relevant emotions from this list: "
                "happiness, sadness, anger, fear, surprise, disgust, love, jealousy, guilt, pride, shame, compassion, sympathy, trust. "
                "For each emotion, assign a score between 0.0 and 1.0.\n"
                "b) Revise the existing answer using the extracted emotions and provided context to produce a final empathetic and emotionally adapted response.\n"
                "c) Provide a detailed explanation of your reasoning behind the revised answer.\n"
                "\n"
                "IMPORTANT: Return ONLY a valid JSON object with exactly these three keys:\n"
                "   'emotional_response' (plain text),\n"
                "   'reasoning' (plain text),\n"
                "   'extracted_emotions' (an array of objects where each object has 'emotion' and 'score', whereby 'score' can only be a value between 0.0 and 1.0).\n"
            )

        # Append dynamic conversation context
        base_prompt += f"\nConversation History: {conversation_history}\n"
        base_prompt += f"User Prompt: {user_prompt}\n"
        base_prompt += f"Average User Emotions: {avg_user_emotions}\n"
        base_prompt += f"Agent's Current Emotional State: {agent_state}\n"
        if llm_answer is not None:
            base_prompt += f"Existing LLM Answer: {llm_answer}\n"

        return base_prompt

    def emotional_response(
        self, 
        user_id: str, 
        prompt: str, 
        user_profile: UserProfile, 
        agent_state: Dict, 
        llm_answer: Optional[str] = None
    ) -> Dict:
        """
        Generate an empathetic response by combining emotion extraction and response generation into one LLM call.
        Expects a JSON response with the following EXACT keys:
            - 'emotional_response': A plain text empathetic answer.
            - 'reasoning': A plain text explanation of your reasoning.
            - 'extracted_emotions': An array of objects, each with 'emotion' and a numeric 'score' between 0.0 and 1.0.
        """
        # Retrieve conversation history and emotional profile
        conversation_history = user_profile.get_conversation_history(10)
        avg_user_emotions = user_profile.get_emotional_profile()

        # Build the combined prompt with full context and detailed instructions
        combined_prompt = self.get_combined_emotional_prompt(
            conversation_history=conversation_history,
            user_prompt=prompt,
            avg_user_emotions=avg_user_emotions,
            agent_state=agent_state,
            llm_answer=llm_answer
        )

        # Send the combined prompt in a single LLM call and expect a JSON response
        raw_response = self.llm.send_prompt(combined_prompt)

        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError:
            # Attempt to sanitize the JSON by escaping newline characters
            sanitized_response = raw_response.replace('\n', '\\n')
            try:
                response_data = json.loads(sanitized_response)
            except json.JSONDecodeError:
                response_data = {
                    "emotional_response": raw_response,
                    "reasoning": "The LLM did not return a valid JSON even after sanitizing newlines.",
                    "extracted_emotions": []
                }

        return response_data
