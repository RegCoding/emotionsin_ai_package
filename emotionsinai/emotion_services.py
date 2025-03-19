import json
import threading
import time
import queue
from typing import Dict, Optional, Tuple, List

from dotenv import load_dotenv

# Assuming BaseLLM, UserProfile, and Response are defined elsewhere in your package.
from .base_llm import BaseLLM
from .user_profile import UserProfile
from .reponse import Response
from .response_split import Response_Split
from .writing_style import WritingStyle
from .reflection import Reflection
from .internal_profile import InternalProfile


class EmotionServices:
    def __init__(self, reflecting: BaseLLM, resource_file_path: str, system_prompt_path: str):
        """
        Initializes the emotion service with two LLM providers and loads an overall emotion setup
        from a JSON file (if available). This new version employs two dedicated threads:
        
          1. reflection_process: waits for a user_id, performs reflection using llm_reflecting,
             and passes a list of tuples to the send_response_process.
             
          2. send_response_process: waits for a list of (string, int) tuples and generates a final response using llm_thinking.
        """
        self.internal_profile = InternalProfile()
        self.internal_profile.load_from_json(resource_file_path)

        self.user_profiles: Dict[str, UserProfile] = {}
        self.llm_reflecting = reflecting      # LLM used for internal reflection
        self.response = Response(llm=self.llm_reflecting)

        # Load the emotion_sytem_prompt from the corresponding json file
        self.emotion_system_prompt = ""  # Initialize the variable
        try:
            with open(system_prompt_path, "r") as file:
                data = json.load(file)
                self.emotion_system_prompt = data.get("emotion_system_prompt", "")  # Correct variable name
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load emotion_system_prompt file '{system_prompt_path}'. Proceeding without emotion setup.")

        # Attributes for storing responses.
        self.new_response = None
        self.processed_reflection = None

        # Set up the two queues:
        # - reflection_queue holds user_id strings.
        # - send_response_queue holds lists of tuples, each tuple a (string, int).
        self.reflection_queue = queue.Queue()         # For reflection_process: input is a user_id.
        self.send_response_queue = queue.Queue()        # For send_response_process: input is a user_id and List[Tuple[str, int]]
        self.input_queue = queue.Queue()  # New queue for input processing

        # Start the two dedicated background threads.
        threading.Thread(target=self.reflection_process, daemon=True).start()
        threading.Thread(target=self.send_response_process, daemon=True).start()
        threading.Thread(target=self.process_input, daemon=True).start()  # New input processing thread

    def get_prompt_extension(self, user_id):
        """
        Returns a prompt extension that includes the current emotional state and profile of the user.
        This is required to ensure that the LLM can generate responses that are emotionally appropriate.
        """

        return f"""{self.emotion_system_prompt}. 
            Your current name:"{self.internal_profile.my_name}";
            Your current goal:"{self.internal_profile.my_goal}";
            Your current role:"{self.internal_profile.my_role}";
            Your current history:"{self.internal_profile.my_history}";
            Your current emotions:"{self.internal_profile.emotional_profile}"; 
            Your current personality traits:"{self.internal_profile.personality_traits}"; 
            Your current motivational drivers:"{self.internal_profile.motivational_drivers}"; 
            Your current ethical framework:"{self.internal_profile.ethical_framework}"; 
            Your current learning behavior:"{self.internal_profile.learning_behavior}"; 
            Your current relationship building:"{self.internal_profile.relationship_building}".
            Your emotions about the user you are just talking to:"{self.get_user_profile(user_id).get_emotional_profile()}".     
        """ 
    def get_new_response(self):
        """
        Checks for a new response from the emotion service.
        Returns the new response if available, or None if not.
        """
        return_response = self.new_response
        self.new_response = None  # Clear the previous response.
        return return_response
    
    def get_self_reflection(self):
        """
        Checks for a new self-reflection from the emotion service.
        Self-reflection is a process where the AI reflects on its own emotional state and behavior.
        Its like an internal log for the AI Agent and typically used for debugging or monitoring purposes.
        Returns the new self-reflection if available, or None if not.
        """
        reflection = self.processed_reflection
        self.processed_reflection = ""  # Clear the previous reflection.
        return reflection
    
    def set_self_reflection(self, new_reflection):
        """
        Set the new self-reflection for the AI Agent.
        """
        self.processed_reflection = new_reflection
    
    def get_self_emotions(self):
        """
        Retrieve the current emotional state of the agent.
        """
        return self.internal_profile.emotional_profile
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """
        Retrieve the user profile for a given user. If the profile does not exist, it is created.
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]
    
    def parse_input(self, user_input: str) -> dict:
        """
        Parse the user input to extract both overall appraisal scores and specific emotion levels.
        
        The LLM is prompted to output a JSON object containing the following keys with scores between 0 and 1:
        - "sentiment_score": Overall sentiment (0: very negative, 1: very positive)
        - "relevance": How relevant the prompt is relative to the context.
        - "novelty": Degree of unexpectedness in the prompt.
        - "goal_alignment": How well the prompt aligns with the agent's goals.
        - "controllability": How manageable or controllable the situation is.
        - "normative_significance": The importance of the prompt based on social norms.
        - "emotion_levels": A nested JSON object with these keys:
                "happiness", "sadness", "anger", "fear", "surprise", "disgust",
                "love", "jealousy", "guilt", "pride", "shame", "compassion",
                "sympathy", "trust".
                
        Returns:
        A dictionary with the extracted keys and their corresponding scores.
        """
        prompt = f"""
            You are an expert NLP analyzer designed to extract detailed emotional and appraisal scores from a user’s input. 
            Please analyze the following user prompt and provide a JSON object that includes the following keys with scores between 0 and 1:
            - "sentiment_score": A value representing the overall sentiment of the text (0 being very negative and 1 being very positive).
            - "relevance": How relevant the prompt is in relation to the current context.
            - "novelty": The degree of unexpectedness or new information in the prompt.
            - "goal_alignment": How well the prompt aligns with the agent's goals.
            - "controllability": A measure of how controllable or manageable the situation described in the prompt is.
            - "normative_significance": The importance of the prompt based on social norms or expected interactions.
            - "emotion_levels": A JSON object containing the following keys, each with a score between 0 and 1:
                "happiness", "sadness", "anger", "fear", "surprise", "disgust", "love", "jealousy", "guilt", "pride", "shame", "compassion", "sympathy", "trust".

            Please ensure that each key is assigned a numerical score between 0 and 1, where 0 means the attribute is absent and 1 means it is at its maximum. 
            Please only return the JSON object, no explanation or other text. And never put any single or double quotation marks before and after the JSON object.

            Here is the user prompt:

            {user_input}
            """
        # Send the prompt to the LLM and capture its response.
        response = self.llm_reflecting.send_prompt(prompt)
        #print(f"LLM response: {response}")

        # Attempt to convert the response to a Python dictionary.
        try:
            result = json.loads(response)
        except Exception as e:
            print("Error parsing JSON:", e)
            result = {}

        return result

    
    def evaluate_appraisal(self, input_scores: dict) -> dict:
        """
        Evaluate the new input using multiple appraisal dimensions based on validated psychological theories,
        such as Lazarus's appraisal theory. The function uses two main appraisal stages:
        
        1. Primary Appraisal: Assessing the significance of the stimulus.
            - Includes relevance, novelty, and normative significance.
        2. Secondary Appraisal: Evaluating coping potential.
            - Includes goal alignment and controllability.
        
        The function then computes an overall appraisal score that can be used to update the emotional state.
        
        Expected keys in input_scores:
        - "sentiment_score": Overall sentiment (0 to 1).
        - "relevance": How relevant the prompt is (0 to 1).
        - "novelty": Degree of unexpectedness (0 to 1).
        - "goal_alignment": Alignment with the agent's goals (0 to 1).
        - "controllability": Perceived controllability (0 to 1; lower scores might indicate more stress).
        - "normative_significance": How well the input fits with expected social norms (0 to 1).
        
        Returns:
        A dictionary containing the following keys:
            - "primary_appraisal": Combined score of stimulus significance.
            - "secondary_appraisal": Combined score of coping potential.
            - "overall_appraisal": Weighted overall score from primary and secondary appraisals.
            - All original scores for traceability.
        """
        # Primary appraisal: Significance of the stimulus.
        # Here, high relevance, novelty, and normative significance imply that the input is very salient.
        primary_weights = {"relevance": 0.4, "novelty": 0.3, "normative_significance": 0.3}
        primary_appraisal = (
            input_scores.get("relevance", 0) * primary_weights["relevance"] +
            input_scores.get("novelty", 0) * primary_weights["novelty"] +
            input_scores.get("normative_significance", 0) * primary_weights["normative_significance"]
        )
        
        # Secondary appraisal: Coping potential.
        # High goal alignment and high controllability indicate that the agent feels capable of handling the input.
        # Note: For controllability, a lower score might indicate a threat (i.e., feeling less in control).
        # We invert the controllability to represent perceived threat for the appraisal.
        inverted_controllability = 1 - input_scores.get("controllability", 0)
        secondary_weights = {"goal_alignment": 0.6, "inverted_controllability": 0.4}
        secondary_appraisal = (
            input_scores.get("goal_alignment", 0) * secondary_weights["goal_alignment"] +
            inverted_controllability * secondary_weights["inverted_controllability"]
        )
        
        # Overall appraisal: Combine primary and secondary appraisals.
        # A simple approach is to average these two dimensions; you might adjust the weighting as needed.
        overall_appraisal = (primary_appraisal + secondary_appraisal) / 2
        
        # Additionally, we can include the sentiment_score to slightly bias the overall evaluation,
        # assuming a more positive sentiment might slightly buffer negative appraisals.
        sentiment_bias = (input_scores.get("sentiment_score", 0) - 0.5) * 0.2  # Scale bias factor
        overall_appraisal += sentiment_bias
        
        # Clamp the overall appraisal between 0 and 1.
        overall_appraisal = max(0, min(1, overall_appraisal))
        
        appraisal = {
            "primary_appraisal": primary_appraisal,
            "secondary_appraisal": secondary_appraisal,
            "overall_appraisal": overall_appraisal,
            # Include the raw input scores for traceability
            "input_scores": input_scores
        }
        
        return appraisal
    
    def update_emotional_state(self, appraisal: dict, input_scores: dict, user_id: str = "default_user"):
        """
        Updates the internal emotional state of the agent based on appraisal results using principles from
        psychological research and appraisal theory. The function takes into account the overall appraisal
        score, the raw sentiment score, and personality factors (e.g., neuroticism) to determine how to adjust
        baseline emotions.

        Assumptions:
        - self.internal_profile is an instance of InternalProfile.
        - self.internal_profile.emotional_profile["baseline_emotions"] is a dictionary with keys such as:
            "happiness", "sadness", "anger", "fear", "surprise", "love", "pride", etc.
        - Personality traits (especially "neuroticism") are defined within
            self.internal_profile.personality_traits["big_five"] with values between 0 and 1.
        
        The updating process works as follows:
        1. Overall appraisal is used to derive two types of adjustments:
            - A positive adjustment if the overall appraisal is above a neutral point (0.5),
            boosting positive emotions and reducing negative ones.
            - A negative adjustment if the appraisal is below the neutral point,
            increasing negative emotions. The impact is amplified by the agent's neuroticism.
        2. Novelty is also considered to modulate the 'surprise' emotion.
        3. The updated values are clamped between 0 and 1 to ensure valid emotion intensities.
        """
        # Retrieve baseline emotions from the internal profile.
        baseline = self.internal_profile.emotional_profile.get("baseline_emotions", {})

        # For demonstration purposes, we assume baseline emotions are already initialized.
        # If a specific emotion is missing, default to a neutral value of 0.5.
        def get_emotion(emotion: str) -> float:
            return baseline.get(emotion, 0.5)
        
        # Get personality factor for neuroticism; higher neuroticism amplifies negative reactions.
        neuroticism = self.internal_profile.personality_traits.get("big_five", {}).get("neuroticism", 0.5)
        
        # Extract the overall appraisal and sentiment score from the appraisal and input_scores.
        overall_appraisal = appraisal.get("overall_appraisal", 0.5)
        sentiment_score = input_scores.get("sentiment_score", 0.5)
        
        # Define adjustment factors.
        # Positive adjustment: if overall_appraisal > 0.5, boost positive emotions.
        positive_adjustment = (overall_appraisal - 0.5) * 0.2
        # Negative adjustment: if overall_appraisal < 0.5, amplify negative emotions, scaled by neuroticism.
        negative_adjustment = (0.5 - overall_appraisal) * 0.2 * (1 + neuroticism)
        
        # Update positive emotions: happiness, love, and pride.
        baseline["happiness"] = min(1.0, max(0.0, get_emotion("happiness") + positive_adjustment))
        baseline["love"] = min(1.0, max(0.0, get_emotion("love") + positive_adjustment))
        baseline["pride"] = min(1.0, max(0.0, get_emotion("pride") + positive_adjustment))
        
        # For negative emotions, if the appraisal is positive, we reduce them.
        baseline["sadness"] = min(1.0, max(0.0, get_emotion("sadness") - positive_adjustment))
        baseline["anger"] = min(1.0, max(0.0, get_emotion("anger") - positive_adjustment))
        baseline["fear"] = min(1.0, max(0.0, get_emotion("fear") - positive_adjustment))
        
        # Conversely, if the overall appraisal is negative, increase negative emotions.
        baseline["sadness"] = min(1.0, max(0.0, get_emotion("sadness") + negative_adjustment))
        baseline["anger"] = min(1.0, max(0.0, get_emotion("anger") + negative_adjustment))
        baseline["fear"] = min(1.0, max(0.0, get_emotion("fear") + negative_adjustment))
        
        # Adjust the 'surprise' emotion based on novelty.
        novelty = input_scores.get("novelty", 0.5)
        baseline["surprise"] = min(1.0, max(0.0, get_emotion("surprise") + (novelty - 0.5) * 0.1))
        
        # Optionally, sentiment_score can further bias the emotions.
        # For example, a higher sentiment score could gently nudge the state towards positivity.
        sentiment_bias = (sentiment_score - 0.5) * 0.05
        baseline["happiness"] = min(1.0, max(0.0, baseline["happiness"] + sentiment_bias))
        
        # Save the updated baseline emotions back into the internal profile.
        self.internal_profile.emotional_profile["baseline_emotions"] = baseline
        
        # Optionally, update user-specific feelings (if such a mechanism exists)
        # For example, you might store an aggregated "feeling towards user" that considers both the updated mood
        # and historical interactions.
        # self.user_feeling[user_id] = baseline["happiness"]  # This is a simplified example.

    def add_input(self, user_id: str, prompt: str, answer: Optional[str] = None, writing_style: bool = False, text_split: bool = False):
        """
        Add a new input to the emotion service queue for processing. The input is added as a tuple containing: 
        - user_id: A unique identifier for the user.
        - prompt: The user's prompt or message.
        - answer: The AI's response or answer (if available).
        - writing_style: A boolean indicating whether to adapt the writing style of the response.
        - text_split: A boolean indicating whether to split the response into multiple parts for a more human-like interaction.
        """
        self.input_queue.put((user_id, prompt, answer, writing_style, text_split))

    def process_input(self):
        """
        Process the input queue by extracting emotional scores from the user input, updating the internal emotional system,
        and adapting the emotional response to the historic writing style if required.
        """
        while True:
            user_id, prompt, answer, writing_style, text_split = self.input_queue.get()
            
            # Extract emotional scores from the user input and update the internal emotional system.
            user_profile = self.get_user_profile(user_id)
            scores = self.parse_input(prompt)
            appraisal = self.evaluate_appraisal(scores)
            self.update_emotional_state(appraisal, scores, user_id)

            self.processed_reflection = "-extract emotional scores from user input and update internal emotional system"
            emotion_levels = scores.get("emotion_levels", {})
            new_emotions = [{"emotion": key, "score": value} for key, value in emotion_levels.items()]
            
            # Add the user's message to the conversation history.
            user_profile.add_message("User", prompt, new_emotions)
            # Updates the user's emotional profile of the ai agent with the new emotions extracted from the last user input.
            user_profile.update_emotions(new_emotions)

            # Optionally adapt the writing style of the response.
            if writing_style:
                writing_style_instance = WritingStyle(self.llm_reflecting)
                adapted_answer = writing_style_instance.adapt_writing_style(user_id, user_profile, new_emotions, answer)
                self.processed_reflection = "-adapt emotional response to the historic writing style"
            else:
                adapted_answer = answer

            # Optionally split the response into multiple parts for a more human-like interaction.
            if text_split:
                response_split = Response_Split(self.llm_reflecting)
                response_list = response_split.return_response_split(user_id, prompt, user_profile, new_emotions, adapted_answer)
                self.processed_reflection = "-split up response into human-like chat interaction"
            else:
                response_list = [(adapted_answer, 0)]

            # Add the response to the send_response_queue for further processing.
            self.send_response_queue.put((user_id, response_list))

    def reflection_process(self):
        """
        Receives a reflection tuple (user_id, text, delay) from the reflection_queue,
        where 'text' is the message (confirmation or reminder) and 'delay' is the time in milliseconds
        after which the message should be sent.
        After the delay, the function sets self.new_response to the text and updates the user's conversation history.
        """
        while True:
            # Wait until a reflection tuple is available.
            user_id, response, delay = self.reflection_queue.get()  # blocking call; expects a tuple (user_id, text, delay)
            # Wait for the specified delay (convert milliseconds to seconds).
            time.sleep(delay / 1000.0)
            # Set the new response.
            self.new_response = response
            # Retrieve the user's profile and update conversation history.
            user_profile = self.get_user_profile(user_id)
            user_profile.add_message("You", self.new_response)
            print(f"[reflection_process] Sent reflection to user {user_id}: {response}")

    def send_response_process(self):
        """
        This thread waits for a tuple (user_id, list_of_tuples) from the send_response_queue.
        Each tuple consists of:
          - A user_id (string)
          - A list of (string, int) tuples
        The function processes the list as follows:
          - It takes the first tuple from the list,
          - Sets self.new_response to the tuple’s string,
          - Sleeps for the number of seconds specified by the integer,
          - Removes that tuple,
          - And continues until the list is empty.
        """
        while True:
            # Wait until a tuple (user_id, list_of_tuples) is available.
            user_id, tuples_list = self.send_response_queue.get()  # blocking call
            backup_responses = tuples_list
            print(f"[send_response_process] Received tuples list for user {user_id}: {tuples_list}")
            user_profile = self.get_user_profile(user_id)
            
            # Process each tuple one-by-one.
            while tuples_list:
                text, delay = tuples_list.pop(0)  # Remove the first tuple.
                self.new_response = text
                
                # Update the user's conversation history.
                user_profile.add_message("You", self.new_response)

                print(f"[send_response_process] Processing tuple for user {user_id}: text='{text}', delay={delay}")
                time.sleep(delay / 200)

            print("START REFLECTION PROCESS.........")
            #Initiate the reflection process
            reflection = Reflection(self.llm_reflecting)
            #print(self.emotion_setup.get("emotional_parameters"))
            #json_emotional_parameters = reflection.self_reflection(self.emotion_setup.get("emotional_parameters"), user_profile, backup_responses)
            #self.set_self_emotions(json_emotional_parameters)
            #rint(f"NEW SELF EMOTIONS:{json_emotional_parameters}")

            #self.processed_reflection = "-update internal emotional system"

            #self.emotion_setup["emotional_parameters"] = json_emotional_parameters
            #print("SELF EMOTIONS UPDATED")

            reflection_response = reflection.set_reminder(user_id, user_profile, tuples_list)
            self.reflection_queue.put((user_id, *reflection_response))
            self.processed_reflection = "-set reminder for future user interaction"
