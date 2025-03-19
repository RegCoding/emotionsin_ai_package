from typing import List, Dict, Optional
import json

from .base_llm import BaseLLM
from .user_profile import UserProfile

class WritingStyle:
    def __init__(self, llm: BaseLLM):  
        self.llm = llm

    def adapt_writing_style(self, user_id: str, user_profile: UserProfile, agent_state: Dict, llm_answer: str) -> str:
        """
        Adapt the given llm_answer to the writing style of the user.
        This adaptation is based on:
          - The recent conversation history with the user.
          - The user's average emotional profile (with trust and sympathy levels in mind).
          - Additional contextual parameters from the agent_state.
        The prompt instructs the LLM to return a final adapted answer that resonates with the user's style.
        """
        # Retrieve the last 10 messages from the user's conversation history.
        conversation_history = user_profile.get_conversation_history(10)
        # Retrieve the user's emotional profile.
        avg_user_emotions = user_profile.get_emotional_profile()
        
        # Build the prompt instructing the LLM how to adapt the answer.
        prompt = (
            "You are a language model expert at adapting written responses to match a user's unique writing style. "
            "Consider the following details:\n\n"
            "Recent Conversation History (last 10 messages): {conversation_history}\n\n"
            "User Emotional Profile: {avg_user_emotions}\n\n"
            "Own inner emotional state: {agent_state}\n\n"
            "Original Answer: {llm_answer}\n\n"
            "Instructions: Adapt the original answer so that it reflects the user's writing style and tone. "
            "Please adapt the writing style very carefully. The full adaption to the user's writing style should only be applied when a very high level of trust and sympathy is reached."
            "Return only the adapted answer as plain text no explanations how you came to the adpated writing style."
        ).format(
            user_id=user_id,
            conversation_history=json.dumps(conversation_history, ensure_ascii=False, indent=2),
            avg_user_emotions=json.dumps(avg_user_emotions, ensure_ascii=False, indent=2),
            agent_state=json.dumps(agent_state, ensure_ascii=False, indent=2),
            llm_answer=llm_answer
        )
        
        # Send the prompt to the LLM to receive the adapted answer.
        adapted_answer = self.llm.send_prompt([{"role": "system", "content": prompt}])
        return adapted_answer
