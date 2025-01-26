# base_llm.py

from abc import ABC, abstractmethod
from typing import Union, List, Dict

class BaseLLM(ABC):
    """
    Abstract base class defining the minimal contract for an LLM provider.
    """

    @abstractmethod
    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """
        Subclasses must implement this method to handle either:
         - A single string prompt
         - A list of chat messages (dicts with 'role' and 'content')
        Return the LLM's response as a string.
        """
        pass
