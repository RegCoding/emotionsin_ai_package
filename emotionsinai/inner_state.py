from typing import List, Dict

class InnerState:
    def __init__(self):
        self.agent_states = {}  # user_id -> { "trust": ..., "empathy": ..., ... }

    def get_or_create_state(self, user_id: str) -> Dict[str, float]:
        if user_id not in self.agent_states:
            # Initialize a default emotional state for the agent
            self.agent_states[user_id] = {
                "trust": 0.5,
                "empathy": 0.5,
                "frustration": 0.0,
                "positivity": 0.5,
                "negativity": 0.0,
                "closeness": 0.3
            }
        return self.agent_states[user_id]

    def update_agent_state_with_multi(
        self, 
        user_id: str, 
        user_emotion_scores: Dict[str, float],
        consistent_factor: float = 0.7
    ):
        """
        Adjust the agent’s emotional state based on the multi-emotion scores from the user.
        
        The 'consistent_factor' emphasizes repeated emotional signals vs. one-off outliers.
        E.g., if user has consistently high frustration, it accumulates in the agent state.
        But if there's just one spike in frustration, maybe we only do a partial shift 
        and become 'curious' instead of drastically updating frustration in the agent.
        """

        state = self.get_or_create_state(user_id)
        # Example logic:

        # Weighted approach to repeated signals:
        # If user_emotion_scores["frustration"] is high but previous frustration average is low,
        # interpret it as possibly one-off. We partially incorporate it.
        frustration_user = user_emotion_scores.get("frustration", 0.5)
        old_frustration_agent = state["frustration"]

        # If there's consistent frustration over time, we increment more
        if frustration_user > 0.6:
            # If repeated (the agent sees a pattern or user_profile average is high?), 
            # increment frustration strongly
            increment = 0.05 if old_frustration_agent < 0.9 else 0.0
            state["frustration"] = min(state["frustration"] + increment, 1.0)
        else:
            # Decrease frustration slightly
            state["frustration"] = max(state["frustration"] - 0.02, 0.0)

        # Trust/positivity logic
        trust_user = user_emotion_scores.get("trust", 0.5)
        positivity_user = user_emotion_scores.get("positivity", 0.5)
        
        # If user consistently expresses trust & positivity, the agent's trust may go up
        # Weighted by consistent_factor if we see it's not just a single one-off.
        adjusted_trust_delta = (trust_user - 0.5) * 0.1 * consistent_factor
        state["trust"] = max(min(state["trust"] + adjusted_trust_delta, 1.0), 0.0)

        # Similarly for positivity or closeness
        closeness_user = user_emotion_scores.get("closeness", 0.5)
        closeness_delta = (closeness_user - 0.5) * 0.05 * consistent_factor
        state["closeness"] = max(min(state["closeness"] + closeness_delta, 1.0), 0.0)

        # If user displays strongly negative outlier, we might degrade trust slightly
        negativity_spike = positivity_user < 0.2 and frustration_user > 0.7
        if negativity_spike:
            state["trust"] = max(state["trust"] - 0.05, 0.0)
            state["empathy"] = max(state["empathy"] - 0.02, 0.0)

        # Another approach:
        #   - If there's a strong mismatch from previous user profile average,
        #     the agent might become "curious" about the cause. Possibly store a "curiosity" field.

    def get_agent_state(self, user_id: str) -> Dict[str, float]:
        return self.get_or_create_state(user_id)
