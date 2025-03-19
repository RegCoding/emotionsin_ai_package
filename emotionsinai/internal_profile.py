import json
from typing import Dict, Any, List

class InternalProfile:
    def __init__(self):
        # Basic agent attributes
        self.my_name: str = ""
        self.my_goal: str = ""
        self.my_role: str = ""
        self.my_history: str = ""
        
        # Personality traits
        self.personality_traits: Dict[str, Any] = {
            "big_five": {},
            "other_traits": {}
        }
        
        # Emotional profile
        self.emotional_profile: Dict[str, Any] = {
            "baseline_emotions": {},
            "emotional_regulation": "",
            "emotional_stability": "",
            "emotional_expression": "",
            "emotional_awareness": "",
            "emotional_triggers": "",
            "emotional_adaptability": "",
            "emotional_authenticity": ""
        }
        
        # Motivational drivers
        self.motivational_drivers: Dict[str, List[str]] = {
            "primary": [],
            "secondary": []
        }
        
        # Ethical framework
        self.ethical_framework: Dict[str, str] = {
            "moral_alignment": "",
            "company_values": ""
        }
        
        # Learning behavior
        self.learning_behavior: Dict[str, str] = {
            "reflection_frequency": "",
            "adaptation_strategy": ""
        }
        
        # Relationship building
        self.relationship_building: Dict[str, str] = {
            "trust_formation_speed": "",
            "collaboration_style": ""
        }
    
    def load_from_json(self, json_str: str) -> None:
        """
        Parses the provided JSON string and populates the class variables.
        """
        try:
            with open(json_str, "r") as file:
                data = json.load(file)
                self.emotion_setup = data.get("emotion_setup", {})

                setup = data.get("emotion_setup", {})
        
                self.my_name = setup.get("my_name", "")
                self.my_goal = setup.get("my_goal", "")
                self.my_role = setup.get("my_role", "")
                self.my_history = setup.get("my_history", "")
                self.my_system_prompt = setup.get("my_system_prompt", "")
                
                self.personality_traits = setup.get("personality_traits", self.personality_traits)
                self.emotional_profile = setup.get("emotional_profile", self.emotional_profile)
                self.motivational_drivers = setup.get("motivational_drivers", self.motivational_drivers)
                self.ethical_framework = setup.get("ethical_framework", self.ethical_framework)
                self.learning_behavior = setup.get("learning_behavior", self.learning_behavior)
                self.relationship_building = setup.get("relationship_building", self.relationship_building)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load resource file '{json_str}'. Proceeding without emotion_setup.")

    def export_to_json(self) -> str:
        """
        Exports the current internal state to a JSON string following the provided template.
        """
        data = {
            "emotion_setup": {
                "my_name": self.my_name,
                "my_goal": self.my_goal,
                "my_role": self.my_role,
                "my_history": self.my_history,
                "my_system_prompt": self.my_system_prompt,
                "personality_traits": self.personality_traits,
                "emotional_profile": self.emotional_profile,
                "motivational_drivers": self.motivational_drivers,
                "ethical_framework": self.ethical_framework,
                "learning_behavior": self.learning_behavior,
                "relationship_building": self.relationship_building
            }
        }
        return json.dumps(data, indent=4)