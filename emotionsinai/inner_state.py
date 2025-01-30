from typing import List, Dict

class InnerState:
    def __init__(self, emotion_setup: Dict):
        self.emotion_setup = emotion_setup.get("emotion_setup", {})
        
        print(f"InnerState class created {self.emotion_setup}")
        print(self.emotion_setup.get("unique_name", ""))
        
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

        print (f"INNER STATE OF THE AGENT: {self.agent_state}")
    
    def get_state(self) -> Dict:
        """Returns the single agent state."""
        return self.agent_state
