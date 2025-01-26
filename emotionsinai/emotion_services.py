from dotenv import load_dotenv
from typing import List, Dict
import json

from emotionsinai import BaseLLM
from emotionsinai import ConversationRepository


class EmotionServices:
    def __init__(self, conversation_repo: ConversationRepository, llm_provider: BaseLLM, resource_file_path: str):
        """
        Inject a ConversationRepository for storing conversation and
        an LLM provider for sending requests to an LLM.
        """
        self.conversation_repo = conversation_repo
        self.llm_provider = llm_provider

        # Load empathy parameters
        empathy_parameters = self.load_empathy_parameters(resource_file_path)
        self.context = empathy_parameters.get("context", "")
        self.motivations = empathy_parameters.get("motivations", [])


    def load_empathy_parameters(self, file_path:str="resources.json"):
        """
        Load empathy parameters (context and motivations) from a JSON file.
        
        :param file_path: Path to the resource file.
        :return: A dictionary containing context and motivations.
        """
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return data.get("empathy_parameters", {})
        except FileNotFoundError:
            raise RuntimeError(f"Resource file '{file_path}' not found.")
        except json.JSONDecodeError:
            raise RuntimeError(f"Resource file '{file_path}' contains invalid JSON.")

    def get_answer(self, user_prompt: str) -> str:
        """
        Calls the LLM provider to get a direct response to the user's prompt (string).
        """
        return self.llm_provider.send_prompt(user_prompt)

    def get_response_and_emotion(self, user_id: str, user_prompt: str):
        """
        1. Retrieve conversation for this user.
        2. Append user’s new message (role='user') to conversation.
        3. Build a list of messages (chat format) for the LLM to see the full context.
        4. Use that to get the main assistant reply.
        5. In parallel, do a separate LLM call for emotion analysis.
        6. Store the user message + emotion, and the assistant reply in the repository.
        7. Return (assistant_reply, emotion).
        """
        # 1) Retrieve existing conversation for user
        conversation_history = self.conversation_repo.get_conversation(user_id)

        # 2) Add the user’s new message to conversation
        conversation_history.append({"role": "user", "content": user_prompt})

        # 3) Build the messages for chat format
        #    You can decide if you want to prepend an explicit system message,
        #    or rely on the provider to prepend if none exists.
        messages_for_chatgpt = conversation_history.copy()

        # 4) Send full context to LLM for the main response
        assistant_reply = self.llm_provider.send_prompt(messages_for_chatgpt)

        # 5) Do a separate call for emotion
        emotion_analysis_prompt = (
            f"Analyze the user's emotion in this input:\n\n{user_prompt}"
        )
        emotion = self.llm_provider.send_prompt(emotion_analysis_prompt)

        # 6) Store user’s message (with emotion) and assistant reply
        self.conversation_repo.add_message(
            user_id=user_id, 
            role="user", 
            content=user_prompt, 
            emotion=emotion
        )
        self.conversation_repo.add_message(
            user_id=user_id,
            role="assistant",
            content=assistant_reply
        )

        return assistant_reply, emotion

    def empathy_assessment(self, user_id, emotion):
        """
        Suggest an empathetic approach. Simpler call with fewer messages.
        """
        prompt = f"""
        Context: {self.context}
        Detected Emotion: {emotion}
        Possible Motivations: {self.motivations}
        Evaluate the user's situation and suggest an empathetic approach 
        that addresses their emotional state and motivations, 
        aligning with our company's values.
        """
        # Here we just send a single string prompt:
        return self.llm_provider.send_prompt(prompt)

    def draft_empathic_response(self, user_id, initial_answer, empathy_assessment, empathy_temperature):
        """
        Combine initial_answer + empathy_assessment into a new empathic reply.
        Use conversation_history so we can mirror user's style if we want.
        """
        conversation_history = self.conversation_repo.get_conversation(user_id)

        system_msg = (
            "You are an empathetic assistant. Use a writing style similar to the user's style. "
            "Please read the conversation so far to match tone and wording."
        )

        prompt = f"""
        The user asked a question, and you gave an answer: {initial_answer}
        The empathy assessment suggests: {empathy_assessment}
        Please combine these into a single empathic response. 
        Mirror the user's tone and style, and adjust empathy level to {empathy_temperature}.
        """

        # Build final messages
        messages_for_chatgpt = [{"role": "system", "content": system_msg}] \
                               + conversation_history + \
                               [{"role": "user", "content": prompt}]

        empathic_reply = self.llm_provider.send_prompt(messages_for_chatgpt)

        # Store the final empathic reply in the conversation
        self.conversation_repo.add_message(
            user_id=user_id,
            role="assistant",
            content=empathic_reply
        )

        return empathic_reply
