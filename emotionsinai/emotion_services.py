import json
import threading
import time
import queue
from typing import Dict, Optional

from dotenv import load_dotenv

# Assuming BaseLLM, UserProfile, and Response are defined elsewhere in your package.
from .base_llm import BaseLLM
from .user_profile import UserProfile
from .reponse import Response


class EmotionServices:
    def __init__(self, thinking: BaseLLM, reflecting: BaseLLM, resource_file_path: str):
        """
        Initializes the emotion service with two LLM providers (one for generating the answer with emotional context,
        and one for internal reflection) and loads the emotional setup from a JSON file.
        Additionally, this class starts background threads for processing input and for internal agent processing.
        """
        self.user_profiles: Dict[str, UserProfile] = {}
        self.llm_thinking = thinking        # LLM to generate the external response
        self.llm_reflecting = reflecting      # Smaller, cheaper LLM for internal reflection
        self.response = Response(llm=self.llm_thinking)

        # Load the full setup, including emotion_setup and internal_agents.
        self.full_setup = self.load_emotion_setup(resource_file_path)
        self.emotion_setup = self.full_setup.get("emotion_setup", {})

        # Set up agent state from the emotion_setup section.
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

        # These attributes are used to store responses.
        self.new_response = None
        self.processed_reflection = None

        # Initialize internal agents (from the JSON) and set up inter-thread communication.
        self.internal_agents: Dict[int, Dict] = {}  # keys: agent id, values: agent config
        self.agent_queues: Dict[int, queue.Queue] = {}  # communication channel per agent id
        self.agent_threads: Dict[int, threading.Thread] = {}  # store thread references

        self.setup_internal_agents(self.full_setup.get("internal_agents", {}))

        # Start additional background tasks (if needed).
        threading.Thread(target=self.process_new_input, daemon=True).start()
        # Start background threads
        self.start_background_tasks()

    def get_new_response(self):
        return self.new_response
    
    def load_emotion_setup(self, file_path: str = "resources.json") -> Dict:
        """
        Load empathy parameters and internal agents from a JSON file.

        :param file_path: Path to the resource file.
        :return: A dictionary containing the full configuration.
        """
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            raise RuntimeError(f"Resource file '{file_path}' not found.")
        except json.JSONDecodeError:
            raise RuntimeError(f"Resource file '{file_path}' contains invalid JSON.")

    def setup_internal_agents(self, agents_config: Dict):
        """
        Sets up the internal agents from the provided configuration.
        For each internal agent, a dedicated queue and thread is created.
        Each agent thread waits for input from the agent with an id equal to its own id - 1.
        For agent with id 1, input is expected to come directly from add_input().
        """
        # Build a mapping from agent id (as integer) to its configuration.
        for key, agent in agents_config.items():
            try:
                agent_id = int(agent.get("id"))
            except (ValueError, TypeError):
                continue  # skip invalid agent ids
            self.internal_agents[agent_id] = agent
            # Create a queue for each agent.
            self.agent_queues[agent_id] = queue.Queue()

        # Create and start a thread for each internal agent.
        for agent_id in sorted(self.internal_agents.keys()):
            agent = self.internal_agents[agent_id]
            thread = threading.Thread(target=self.internal_agent_worker, args=(agent,), daemon=True)
            thread.start()
            self.agent_threads[agent_id] = thread

    def internal_agent_worker(self, agent: Dict):
        """
        Worker function for an internal agent.
        Each agent waits for input from its predecessor's queue (agent_id - 1).
        For agent 1, the input is expected to be placed by add_input().
        The worker processes the message (here simulated by a simple transformation)
        and then, if there is a next agent in the chain, passes the processed output along.
        If it is the last agent, the output is stored in self.processed_reflection.
        """
        agent_id = int(agent.get("id"))
        agent_name = agent.get("name", f"agent_{agent_id}")
        while True:
            # Determine which queue to listen on.
            if agent_id == 1:
                # The first agent receives input directly from add_input().
                input_message = self.agent_queues[1].get()  # blocking call
            else:
                # For other agents, wait on the queue of the previous agent.
                input_message = self.agent_queues[agent_id - 1].get()

            #print(f"[Internal Agent {agent_id} - {agent_name}] received input: {input_message}")

            # Simulate processing (this could be replaced by a call to self.llm_reflecting).
            agent_goal = agent.get("goal")
            agent_action = agent.get ("action") + f" - This is required user input, your own emotional state and the current status of the ongoing discussion with other agents: {input_message}"

            messages = [
                {"role": agent_goal, "content": agent_action}
            ]

            agent_answer = self.llm_reflecting.send_prompt(messages)
            processed_message = input_message + f"--> Response {agent_name}: {agent_answer}"
            print(f"[Internal Agent {agent_id} - {agent_name}] processed output: {agent_answer}")

            # Check if there is an agent with id = agent_id + 1.
            if (agent_id + 1) in self.agent_queues:
                # Pass the processed message to the next agent's queue.
                self.agent_queues[agent_id].put(processed_message)
            else:
                # If this is the last agent, store the result.
                self.processed_reflection = processed_message
                #print(f"[Internal Agent {agent_id} - {agent_name}] final processed output: {processed_message}")

            # Brief sleep to simulate processing time.
            time.sleep(1)

    def get_user_profile(self, user_id: str) -> UserProfile:
        """
        Retrieve the user profile for a given user. If not available, create a new one.
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]

    def add_input(self, user_id: str, prompt: str, answer: Optional[str] = None):
        """
        Public method to add new input. This generates a new external response,
        and then injects that response into the processing pipeline by placing it in agent 1's queue.
        """
        user_profile = self.get_user_profile(user_id)
        self.new_response = self.response.emotional_response(user_id, prompt, user_profile, self.agent_state, answer)
        print(f"Intuitive response: {self.new_response}")

        agent_input = f"""This is your current emotional profile about the user: {user_profile}, 
                      the last user prompt: {prompt}, and the generated response: {self.new_response},
                      and the current emotional state of the agent: {self.agent_state}"""
        # Put the new response into the queue for agent 1.
        if 1 in self.agent_queues:
            self.agent_queues[1].put(agent_input)
        else:
            print("[Main] Warning: No internal agent with id 1 found.")

    def process_new_input(self):
        """
        An example background function that could process new input if needed.
        (This function may become obsolete with the introduction of the internal agents pipeline.)
        """
        while True:
            time.sleep(2)
            if self.new_response is not None:
                #print(f"[Background] Processing new input: {self.new_response}")
                # Additional processing logic could go here.
                self.new_response = None

    def start_background_tasks(self):
        """Starts the background processing tasks."""
        threading.Thread(target=self.process_new_input, daemon=True).start()
