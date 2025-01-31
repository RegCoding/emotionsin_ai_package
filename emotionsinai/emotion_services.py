from dotenv import load_dotenv
from typing import List, Dict, Optional
import json
import threading
import time

from .base_llm import BaseLLM
from .user_profile import UserProfile
from .reponse import Response


class EmotionServices:
    def __init__(self, thinking: BaseLLM, reflecting: BaseLLM, resource_file_path: str):
        """
        Inject a ConversationRepository for storing conversation and
        an LLM provider for sending requests to an LLM.
        """
        self.user_profiles: Dict[str, UserProfile] = {}

        self.llm_thinking = thinking        #we use this llm to create the answer to the user prompt including the emotional context
        self.llm_reflecting = reflecting    #this can be a smaller, cheaper llm to support the internal reflection process
        
        self.response = Response(llm=self.llm_thinking)

        # Load empathy parameters and create emotional agent
        self.emotion_setup = self.load_emotion_setup(resource_file_path)   

        self.agent_state = {
            "name": self.emotion_setup.get("my_name", ""),
            "context": self.emotion_setup.get("context", ""),
            "goal": self.emotion_setup.get("my_goal", ""),
            "guardrails": self.emotion_setup.get("guardrails", []),
            "emotional_intensity": self.emotion_setup.get("emotional_parameters", {}).get("emotional_intensity", ""),
            "emotions": self.emotion_setup.get("emotional_parameters", {}).get("emotional_range", {}),
            "emotional_regulation": self.emotion_setup.get("emotional_parameters", {}).get("emotional_regulation", ""),
            "emotional_stability": self.emotion_setup.get("emotional_parameters", {}).get("emotional_stability", ""),
            "emotional_empathy": self.emotion_setup.get("emotional_parameters", {}).get("emotional_empathy", ""),
            "emotional_expression": self.emotion_setup.get("emotional_parameters", {}).get("emotional_expression", ""),
            "emotional_awareness": self.emotion_setup.get("emotional_parameters", {}).get("emotional_awareness", ""),
            "emotional_triggers": self.emotion_setup.get("emotional_parameters", {}).get("emotional_triggers", ""),
            "emotional_adaptability": self.emotion_setup.get("emotional_parameters", {}).get("emotional_adaptability", ""),
            "emotional_authenticity": self.emotion_setup.get("emotional_parameters", {}).get("emotional_authenticity", ""),
            "social_emotional_connectivity": self.emotion_setup.get("emotional_parameters", {}).get("social_emotional_connectivity", ""),
            "cultural_influence_on_emotion": self.emotion_setup.get("emotional_parameters", {}).get("cultural_influence_on_emotion", "")
        }

        self.new_response = None
        self.processed_reflection = None

        # Start background threads
        self.start_background_tasks()


    # New or updated methods to manage user profiles
    def get_user_profile(self, user_id) -> UserProfile:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]
    

    def get_processed_reflection(self):
        return self.processed_reflection
    

    def get_new_response(self):
        return self.new_response
    

    def load_emotion_setup(self, file_path:str="resources.json") -> Dict:
        """
        Load empathy parameters (context and motivations) from a JSON file.
        
        :param file_path: Path to the resource file.
        :return: A dictionary containing context and motivations.
        """
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return data.get("emotion_setup", {})
        except FileNotFoundError:
            raise RuntimeError(f"Resource file '{file_path}' not found.")
        except json.JSONDecodeError:
            raise RuntimeError(f"Resource file '{file_path}' contains invalid JSON.")
        

    def add_input(self, user_id: str, prompt: str, answer: Optional[str] = None):
        """Reflects on the agent's state."""
        user_profile = self.get_user_profile(user_id)
        self.new_response = self.response.emotional_response(user_id, prompt, user_profile, self.agent_state, answer)

        #raw_response = self.reflecting_llm.send_prompt("Hat das funktioniert?")
        #print(f"OUTPUT FROM LLAMA3:{raw_response}")
    

    def process_new_input(self):
        """Background function that processes new input if available."""
        while True:
            time.sleep(2)
            if self.new_response is not None:
                print(f"Processing new input: {self.new_response}")
                self.processed_reflection = "test"
                self.new_response = None
            
    

    def wait_for_processed_output(self):
        """Background function that waits for processed output."""
        while True:
            if self.processed_reflection is not None:
                print(f"Processed output available: {self.processed_reflection}")
                self.processed_reflection = None
            time.sleep(1)
    

    def start_background_tasks(self):
        """Starts the background processing tasks."""
        threading.Thread(target=self.process_new_input, daemon=True).start()
        threading.Thread(target=self.wait_for_processed_output, daemon=True).start()
        


        

