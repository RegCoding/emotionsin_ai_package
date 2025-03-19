from typing import List, Dict, Optional, Tuple
import json

from .base_llm import BaseLLM
from .user_profile import UserProfile

class Reflection:
    def __init__(self, llm: BaseLLM):
        """
        Initializes the Reflection instance with a LLM model and a dictionary of user profiles.
        """
        self.llm = llm

import json
import re
from typing import List, Tuple

class Reflection:
    def __init__(self, llm):
        self.llm = llm

    def self_reflection(self, self_emotions: dict, user_profile, response_list: List[Tuple[str, int]]) -> dict:
        """
        Reflects on recent interactions and updates self-emotions accordingly.

        Ensures that only a valid JSON structure is returned.
        """
        try:
            conversation_history = user_profile.get_conversation_history(10)
            emotional_profile = user_profile.get_emotional_profile()

            prompt = (
                "You are an advanced AI model that updates its emotional profile based on interactions. "
                "Analyze the AI's current self-emotions, user emotional profile, and conversation history. "
                "Return ONLY the updated self_emotions in JSON format and NOTHING else.\n\n"
                "Current AI self_emotions:\n{self_emotions}\n\n"
                "User Emotional Profile:\n{emotional_profile}\n\n"
                "Recent Conversation History:\n{conversation_history}\n\n"
                "Response List (AI responses and user reactions):\n{response_list}\n\n"
                "IMPORTANT: Return ONLY a valid JSON object with updated self_emotions. No explanations, no text, just JSON."
            ).format(
                self_emotions=json.dumps(self_emotions, indent=2),
                emotional_profile=json.dumps(emotional_profile, indent=2),
                conversation_history=json.dumps(conversation_history, indent=2),
                response_list=json.dumps(response_list, indent=2)
            )

            # Get response from LLM
            print(f"REFLECTION PROMPT:\n{prompt}")
            llm_output = self.llm.send_prompt([{"role": "system", "content": prompt}])
            print(f"[Reflection] LLM output:\n{llm_output}")

            # Extract only JSON using regex (handles cases where LLM returns text before/after JSON)
            json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if json_match:
                json_cleaned = json_match.group(0)  # Extract only the JSON part
                updated_emotions = json.loads(json_cleaned)
                return updated_emotions

            print("[Reflection] Error: LLM response did not contain valid JSON")
            return self_emotions  # Return original emotions if no valid JSON is found

        except Exception as e:
            print(f"[Reflection] Unexpected Error: {e}")
            return self_emotions  # Return existing emotions to prevent crashing


    def set_reminder(self, user_id: str, user_profile: UserProfile, response_list: List[Tuple[str, int]]) -> Tuple[str, int]:
        """
        This function evaluates using a LLM model whether it is necessary to send a reminder or a confirmation
        in case the user does not respond to the last message. It evaluates three things:
        
        1. Is it – depending on the emotional profile of the user and the last AI response – necessary to send a reminder at all?
        2. If yes, what should be the content of the reminder? Should it be a confirmation of the last message, or a reminder of it?
        3. What is the best timing to send the reminder?
        
        In case it is appropriate to send a reminder, the function returns, in JSON format, a tuple consisting of a string
        (the reminder message) and an integer (the delay in milliseconds). The JSON result is converted into a Tuple[str, int] and returned.
        """
        # Retrieve the user's conversation history.
        if user_profile is None:
            raise ValueError("User profile not found.")

        conversation_history = user_profile.get_conversation_history(10)
        emotional_profile = user_profile.get_emotional_profile()
        
        # Assume the last AI response is the text part of the last tuple in response_list.
        last_ai_response = response_list[-1][0] if response_list else "No previous AI response."

        # Build the prompt that instructs the LLM how to evaluate the need for a reminder.
        prompt = (
            "You are a highly skilled assistant tasked with determining whether a reminder or confirmation message "
            "should be sent to a user who has not responded to the last message from an AI. Evaluate the following:\n\n"
            "1. The recent conversation history of the user:\n{conversation_history}\n\n"
            "2. The user's emotional profile:\n{emotional_profile}\n\n"
            "3. The last response from the AI:\n{last_ai_response}\n\n"
            "Based on these, decide if it is necessary to send a reminder or confirmation. "
            "If a message is needed, determine the optimal content and timing. "
            "Return your decision as a JSON array containing exactly one object with the following keys:\n"
            "  - 'text': a string with the reminder (or confirmation) message, and\n"
            "  - 'delay': an integer representing the time in milliseconds when the message should be sent.\n"
            "If no message is necessary, return an empty JSON array."
        ).format(
            conversation_history=json.dumps(conversation_history, ensure_ascii=False, indent=2),
            emotional_profile=json.dumps(emotional_profile, ensure_ascii=False, indent=2),
            last_ai_response=last_ai_response
        )

        print(f"REFLECTION PROMPT:{prompt}")
        # Use the LLM to get its evaluation.
        llm_output = self.llm.send_prompt([{"role": "system", "content": prompt}])
        print(f"[Reflection] LLM output: {llm_output}")

        # Parse the JSON output.
        try:
            data = json.loads(llm_output)
            # If the array is empty, no reminder is needed.
            if not data:
                return ("", 0)
            # Otherwise, expect exactly one object.
            obj = data[0]
            text = str(obj["text"])
            delay = int(obj["delay"])
            return (text, delay)
        except Exception as e:
            print(f"[Reflection] Error parsing LLM output: {e}")
            return ("", 0)

