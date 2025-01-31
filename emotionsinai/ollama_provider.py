import ollama
from typing import Union, List, Dict

from emotionsinai import BaseLLM

class OllamaProvider(BaseLLM):
    """
    Implementation of BaseLLM that connects to a llama3.1 model using Ollama.
    """
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """
        Sends a prompt to the Llama3.1 model via Ollama and returns the response.
        """
        if isinstance(prompt, list):
            # Convert chat messages into Ollama's expected format
            formatted_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in prompt])
        else:
            formatted_prompt = prompt
        
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": formatted_prompt}])

        return response.get("message", {}).get("content", "")
