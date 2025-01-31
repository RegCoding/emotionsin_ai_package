from typing import List, Dict, Optional
import json

from .base_llm import BaseLLM
from .user_profile import UserProfile

class WritingStyle:
    def __init__(self, llm: BaseLLM):  

        self.llm = llm