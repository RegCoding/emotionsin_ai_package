# openai_provider.py

from openai import OpenAI
from typing import Union, List, Dict

#from base_llm import BaseLLM
from emotionsinai import BaseLLM


class OpenAIProvider(BaseLLM):
    """
    Default LLM provider using OpenAI's API.
    Allows passing either a single string prompt or a list of message dicts.
    """

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, openai_key: str = ""):
        self.model_name = model_name
        self.temperature = temperature
        # Configure the OpenAI client
        OpenAI.api_key = openai_key
        self.client = OpenAI(api_key=openai_key)

    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """
        If `prompt` is a string, wrap it in a minimal chat message.
        If `prompt` is a list of dicts, we pass it directly to the chat endpoint.
        """
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            # We assume it's already a list of {"role": ..., "content": ...} dicts
            # Optionally, prepend a system message if you always want a system prompt:
            if not any(msg["role"] == "system" for msg in prompt):
                # Prepend a default system message if none found
                prompt = ([{"role": "system", "content": "You are a helpful assistant."}] 
                          + prompt)
            messages = prompt

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )

        return response.choices[0].message.content
