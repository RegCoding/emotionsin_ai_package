from typing import List, Dict, Tuple, Optional
import json

from .base_llm import BaseLLM
from .user_profile import UserProfile

class Response_Split:
    def __init__(self, llm: BaseLLM):  
        self.llm = llm

    def return_response_split(
        self,
        user_id: str,
        prompt: str,
        user_profile: UserProfile,
        agent_state: Dict,
        llm_answer: str
    ) -> List[Tuple[str, int]]:
        """
        Analyze the empathic answer and split it up into pieces so that the overall response appears more human-like.
        The LLM is instructed to return a JSON array where each element is an object with two keys:
          - 'text': a portion of the answer to be sent to the user.
          - 'delay': an integer in milliseconds representing the time interval to wait before sending this piece.
        
        The function converts the JSON output into a list of (text, delay) tuples.
        """
        # Build the prompt for splitting the answer
        response_split_prompt = (
            "You are an assistant that formats long responses into smaller, human-like conversation pieces. "
            "Your task is to split the provided long answer into multiple parts. The split-up should be in a style how humans naturally answer. For each part, "
            "create a JSON object with two keys: 'text' and 'delay'. 'text' should contain the text piece, "
            "and 'delay' should be an integer representing the delay in milliseconds to simulate a human-like pause before sending this piece. "
            "Return only a valid JSON array of such objects without any extra commentary.\n\n"
            f"The answer that should be split up: {llm_answer}\n\n"
            "Return the JSON array now."
        )

        # Call the LLM with the constructed prompt
        llm_output = self.llm.send_prompt([{"role": "system", "content": response_split_prompt}])
        #print(f"[return_response_split] LLM output: {llm_output}")

        # Attempt to parse the JSON output and convert it to a list of (text, delay) tuples.
        try:
            data = json.loads(llm_output)
            # Ensure the parsed data is a list of dictionaries with "text" and "delay" keys.
            result: List[Tuple[str, int]] = [
                (str(item["text"]), int(item["delay"])) for item in data if "text" in item and "delay" in item
            ]
            return result
        except Exception as e:
            print(f"[return_response_split] Error parsing JSON output: {e}")
            return []

