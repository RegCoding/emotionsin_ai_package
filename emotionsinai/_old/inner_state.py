from typing import List, Dict, Optional
import threading
import time

from .base_llm import BaseLLM

class InnerState:
    def __init__(self, emotion_setup: Dict, reflecting_llm: BaseLLM):  
        self.emotion_setup = emotion_setup.get("emotion_setup", {})
        
        #print(f"InnerState class created {self.emotion_setup}")
        #print(self.emotion_setup.get("unique_name", ""))

        self.reflecting_llm = reflecting_llm
        
        self.agent_state = {
            "name": self.emotion_setup.get("unique_name", ""),
            "context": self.emotion_setup.get("context", ""),
            "goal": self.emotion_setup.get("goal", ""),
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

        self.new_input = None
        self.processed_output = None

        # Start background threads
        self.start_background_tasks()


    def get_state(self) -> Dict:
        """Returns the single agent state."""
        return self.agent_state
    
    def add_input(self, prompt: str, answer: str, emotions: Optional[Dict[str, float]]):
        """Reflects on the agent's state."""
        self.new_input = {
            "prompt": prompt,
            "answer": answer,
            "emotions": emotions
        }

        raw_response = self.reflecting_llm.send_prompt("Hat das funktioniert?")
        print(f"OUTPUT FROM LLAMA3:{raw_response}")
    
    def process_new_input(self):
        """Background function that processes new input if available."""
        while True:
            if self.new_input is not None:
                print(f"Processing new input: {self.new_input}")
                self.processed_output = f"Processed: {self.new_input}"
                self.new_input = None
            time.sleep(1)
    
    def wait_for_processed_output(self):
        """Background function that waits for processed output."""
        while True:
            if self.processed_output is not None:
                print(f"Processed output available: {self.processed_output}")
                self.processed_output = None
            time.sleep(1)
    
    def start_background_tasks(self):
        """Starts the background processing tasks."""
        threading.Thread(target=self.process_new_input, daemon=True).start()
        threading.Thread(target=self.wait_for_processed_output, daemon=True).start()
    
    def get_full_answer_prompt(self):
        prompt = (
                "You are an empathetic AI agent responding to a user. Consider the following inputs to craft your response:\n"
                "1. Conversation History: This helps you understand the broader context of the user's prompt. Does the prompt align with previous discussions? Is there any inconsistency in style or content?\n"
                "2. User's Extracted Emotions: These represent how the AI perceives the user's emotions. These feelings naturally influence how the AI should respond.\n"
                "3. Agent's Current Emotional State: Just like a human, the AI's emotional state influences its response.\n"
                "\n"
                "Given this information, formulate a response that a human would naturally give, considering their own state of mind and these contextual factors.\n"
                "Ensure the response is empathetic, contextually relevant, and aligns with the AI's role as a helpful and emotionally aware colleague.\n"
            )
        return prompt
    
    def get_overleigh_answer_prompt(self):
        prompt = (
                "You are an empathetic AI agent responding to a user. Consider the following inputs to craft your response:\n"
                "1. Conversation History: This helps you understand the broader context of the user's prompt. Does the prompt align with previous discussions? Is there any inconsistency in style or content?\n"
                "2. User's Extracted Emotions: These represent how the AI perceives the user's emotions. These feelings naturally influence how the AI should respond.\n"
                "3. Agent's Current Emotional State: Just like a human, the AI's emotional state influences its response.\n"
                "4. Existing LLM Answer: This is a factual response that lacks emotional awareness. Your task is to revise it to accurately reflect the emotional background and context.\n"
                "\n"
                "Given this information, formulate a response that a human would naturally give, considering their own state of mind and these contextual factors.\n"
                "Ensure the response is empathetic, contextually relevant, and aligns with the AI's role as a helpful and emotionally aware colleague.\n"
            )
        return prompt
