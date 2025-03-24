from typing import List, Dict, Optional, Tuple
import json
from .user_profile import UserProfile
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

class Reflection:

    def __init__(self, llm: ChatOllama):
        """
        Initializes the Reflection instance with an LLM model.
        """
        self.llm = llm

    def generate_emotional_guideline(self, user_profil: UserProfile, num_messages: int = 20) -> str:
        """
        Analyzes the user's recent conversation history and emotional profile to generate a
        short, psychologically informed guideline for how the AI should interact with the user
        on an emotional level.

        The output is a 1–2 sentence string focused solely on emotional and psychological handling.

        Args:
            llm: An instance of an LLM interface with a `.send_prompt()` method.
            num_messages: How many recent messages to analyze (default: 20).
        
        Returns:
            A concise emotional interaction guideline string.
        """
        import json

        # Gather relevant context
        conversation_history = user_profil.get_conversation_history(num_messages)

        prompt = (
            "You are a psychological assistant embedded in an AI system. "
            "Your task is to analyze a user's recent conversation history and emotional profile "
            "in order to generate a brief 1–2 sentence guideline for how the AI should respond to this user "
            "in an emotionally intelligent and psychologically safe way.\n\n"
            "Focus only on emotional and psychological best practices based on the user's behavior. "
            "This is NOT about topic content, only about *how* to relate to the user emotionally.\n\n"
            "Recent Conversation History (last few interactions):\n"
            f"{json.dumps(conversation_history, indent=2)}\n\n"
            "Please return ONLY the guideline string. No preamble, no formatting, no JSON. Just the raw guideline text."
        )

        # Send to LLM and return result
        #print("REFLECTION PROMPT SENT TO LLM:\n", prompt)
        response = self.llm.invoke(prompt)
        #print("REFLECTION RESPONSE RECEIVED FROM LLM:\n", response.content)
        guideline = response.content

        #print("###########REFLECTION GUIDELINE GENERATED:", guideline)

        # Update internal guideline
        user_profil.set_guideline(guideline.strip())
        return guideline.strip()


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

        #print(f"REFLECTION PROMPT:{prompt}")
        # Use the LLM to get its evaluation.
        llm_output = self.llm.invoke([{"role": "system", "content": prompt}])
        #print(f"[Reflection] LLM output: {llm_output}")

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

