from dotenv import load_dotenv
from typing import List, Dict
import json

from emotionsinai import BaseLLM
from emotionsinai import ConversationRepository


class EmotionServices:
    def __init__(self, conversation_repo: ConversationRepository, llm_provider: BaseLLM, resource_file_path: str):
        """
        Inject a ConversationRepository for storing conversation and
        an LLM provider for sending requests to an LLM.
        """
        self.conversation_repo = conversation_repo
        self.llm_provider = llm_provider

        # Load empathy parameters
        self.emotion_setup = self.load_emotion_setup(resource_file_path)
        self.context = self.emotion_setup.get("context", "")
        self.goal = self.emotion_setup.get("goal", "")
        self.guardrails = self.emotion_setup.get("guardrails", [])
        self.emotion_parameters = self.emotion_setup.get("emotional_parameters", {})
        self.emotional_intensity = self.emotion_parameters.get("emotional_intensity", "")
        self.emotions = self.emotion_parameters.get("emotional_range", [])
        self.emotional_regulation = self.emotion_parameters.get("emotional_regulation", "")
        self.emotional_stability = self.emotion_parameters.get("emotional_stability", "")
        self.emotional_empathy = self.emotion_parameters.get("emotional_empathy", "")
        self.emotional_expression = self.emotion_parameters.get("emotional_expression", "")
        self.emotional_awareness = self.emotion_parameters.get("emotional_awareness", "")
        self.emotional_triggers = self.emotion_parameters.get("emotional_triggers", "")
        self.emotional_adaptability = self.emotion_parameters.get("emotional_adaptability", "")
        self.emotional_authenticity = self.emotion_parameters.get("emotional_authenticity", "")
        self.social_emotional_connectivity = self.emotion_parameters.get("social_emotional_connectivity", "")
        self.cultural_influence_on_emotion = self.emotion_parameters.get("cultural_influence_on_emotion", "")

        self.user_emotion_scores = {}


    def load_emotion_setup(self, file_path:str="resources.json"):
        """
        Load empathy parameters (context and motivations) from a JSON file.
        
        :param file_path: Path to the resource file.
        :return: A dictionary containing context and motivations.
        """
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return data.get("emotion_parameters", {})
        except FileNotFoundError:
            raise RuntimeError(f"Resource file '{file_path}' not found.")
        except json.JSONDecodeError:
            raise RuntimeError(f"Resource file '{file_path}' contains invalid JSON.")
        

    def get_answer(self, user_prompt: str) -> str:
        """
        Calls the LLM provider to get a direct response to the user's prompt (string).
        """
        return self.llm_provider.send_prompt(user_prompt)


    def assess_llm_response(self, user_prompt: str, llm_response: str, user_id: str) -> Dict:
        """
        Analyze the LLM response to the user's prompt in the context of:
        1) The full conversation history (for emotional and style context).
        2) The AI agent's goal and guardrails.
        3) Potential hidden or emotional aspects that may not be obvious 
            from the prompt alone.

        Returns a dictionary with:
        - 'fit_score': numeric or descriptive rating of how well the answer fits
        - 'improvement_areas': text describing potential improvements
        - 'hidden_emotional_points': any hidden concerns or emotional triggers
        - 'recommendations': a short set of next steps or suggestions 
                            for improving empathy
        """

        # 1) Retrieve existing conversation (excluding this final prompt & answer if you prefer)
        conversation_history = self.conversation_repo.get_conversation(user_id)

        # 2) Extract multi-emotion scores from the user's new message
        self.user_emotion_scores = self.extract_emotion_scores(user_prompt)

        # 3) Prepare a system message that instructs the LLM to perform meta-analysis
        system_msg = (
            "You are a meta-analysis assistant. Your job is to review the conversation "
            "history, the user's latest prompt, and the LLM's response, then critique "
            "the answer for emotional fit, style coherence, and potential hidden aspects "
            "that might have been missed."
        )

        # 3) Build a user message that includes:
        #    - AI's goal
        #    - AI's guardrails
        #    - conversation history
        #    - the latest user prompt
        #    - the LLM's response
        #    - instructions to return a JSON structure
        conversation_text = ""
        for entry in conversation_history:
            conversation_text += f"Role: {entry['role']}\nContent: {entry['content']}\n\n"

        meta_prompt = f"""
        AI Goal: {self.goal}
        Guardrails: {self.guardrails}

        --- Conversation History Start ---
        {conversation_text}
        --- Conversation History End ---

        Latest user question: {user_prompt}
        LLM Response: {llm_response}

        1) Check if and how much the LLM response aligns with the user's emotional state 
        and writing style from the conversation history. 
        2) Identify any potential hidden concerns or emotional triggers the user might have 
        that are not obvious from the user_prompt alone but appear in the conversation history.
        3) Evaluate how well the LLM response fits the AI agent's goal and guardrails.
        4) Suggest how to improve the response in terms of empathy, emotional alignment, or style.

        Return your analysis in valid JSON format with the following keys:
        "fit_score": A numeric or textual rating for how well the response fits (0-10, or Low/Medium/High),
        "improvement_areas": A short text describing what could be improved,
        "hidden_emotional_points": A summary of any deeper emotional aspects not addressed,
        "recommendations": Next steps or suggestions for making the response more empathetic.
        """

        # 4) Construct messages and send to the LLM
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": meta_prompt},
        ]
        analysis_response = self.llm_provider.send_prompt(messages)

        # 5) Parse the LLM's JSON output. If parsing fails, return fallback data.
        try:
            # The LLM might return something like:
            # {
            #   "fit_score": "7/10",
            #   "improvement_areas": "The style is too formal compared to the user's tone.",
            #   "hidden_emotional_points": "User might be anxious about time constraints.",
            #   "recommendations": "Acknowledge user concerns more explicitly, keep a friendlier tone."
            # }
            analysis_dict = json.loads(analysis_response.strip())
        except json.JSONDecodeError:
            # If the LLM didn't return valid JSON, we can fallback or raise an error.
            analysis_dict = {
                "fit_score": "N/A",
                "improvement_areas": "Could not parse the LLM's analysis response.",
                "hidden_emotional_points": "",
                "recommendations": ""
            }

        # Optionally, store the meta-analysis in conversation history or logs:
        # self.conversation_repo.add_message(user_id, "assistant", analysis_response)

        return analysis_dict


    def extract_emotion_scores(self, user_message: str) -> Dict[str, float]:
        """
        Calls the LLM to get multiple emotion scores (0.0 - 1.0) for trust, frustration, positivity, closeness, etc.
        Returns a dict with the scores.
        """
        # This system message instructs the LLM to act as an emotion analyzer
        system_msg = "You are an emotion analysis assistant that outputs JSON with numeric scores for each emotion."

        # For example, define a set of emotions you want:
        user_content = f"""
        Analyze this text and rate the following emotions from 0.0 to 1.0:
            happiness,
            sadness,
            anger,
            fear,
            surprise,
            disgust
            love,
            jealousy,
            guilt,
            pride,
            sympathy,
            trust

        Text: {user_message}

        Return valid JSON of the form:
        {{
        "happiness": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "love": 0.0,
        "jealousy": 0.0,
        "guilt": 0.0,
        "pride": 0.0,
        "sympathy": 0.0,
        "trust": 0.0
        }}
        """

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]

        raw_response = self.llm_provider.send_prompt(messages)

        # Attempt to parse the JSON
        try:
            emotion_scores = json.loads(raw_response)
            # Validate that all required keys exist
            for key in ["trust", "frustration", "positivity", "closeness"]:
                if key not in emotion_scores:
                    emotion_scores[key] = 0.5  # or some default
            return emotion_scores
        except json.JSONDecodeError:
            # fallback in case the LLM doesn't return valid JSON
            return {
                "happiness": 0.5,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "disgust": 0.0,
                "love": 0.0,
                "jealousy": 0.0,
                "guilt": 0.0,
                "pride": 0.0,
                "sympathy": 0.5,
                "trust": 0.0
            }
        

    def draft_empathic_response(
        self, 
        user_id: str, 
        initial_answer: str, 
        analysis_dict: dict
    ) -> str:
        """
        Combine the initial_answer, meta-analysis (analysis_dict), user emotional profile,
        and any outlier info into a final empathic reply. Also mirrors the user's style if desired.
        """

        # 1) Retrieve conversation history
        conversation_history = self.conversation_repo.get_conversation(user_id)

        # 2) Retrieve or create user profile
        user_profile = self.conversation_repo.get_user_profile(user_id)

        # 3) Check for outliers or update profile if no outliers
        outliers_list = user_profile.detect_outliers(self.user_emotion_scores, threshold=0.3)
        if outliers_list:
            outliers = True
            outlier_info = (
                f"The following emotion outliers were detected: {outliers_list}. "
                "Consider showing curiosity or asking clarifying questions if these changes "
                "are unexpected or abrupt.\n"
            )
        else:
            user_profile.update_emotions(self.user_emotion_scores)
            outlier_info = ""

        # 4) Extract details from analysis_dict
        fit_score = analysis_dict.get("fit_score", "N/A")
        improvement_areas = analysis_dict.get("improvement_areas", "")
        hidden_points = analysis_dict.get("hidden_emotional_points", "")
        recommendations = analysis_dict.get("recommendations", "")

        # 5) Gather current rolling averages from the user_profile
        profile_averages = user_profile.rolling_averages  # e.g., {"trust": 0.63, "frustration": 0.15, ...}

        # 6) Create a system message that sets the overall instructions
        system_msg = (
            "You are an empathetic assistant. Use a writing style similar to the user's style. "
            "Please read the conversation so far to match tone and wording. "
            "Incorporate the identified improvements from the meta-analysis into your final response."
        )

        # 7) Build the user prompt that includes all relevant data
        prompt = f"""
        The user asked a question, and you gave an answer: {initial_answer}

        The meta-analysis provided the following insights about this initial answer:
        - Fit Score: {fit_score}
        - Improvement Areas: {improvement_areas}
        - Hidden Emotional Points: {hidden_points}
        - Recommendations: {recommendations}

        Your own emotional profile (rolling averages) of the user you are talking to is:
        {profile_averages}
        Your emotional profile directly impacts the way you respond to the user. 
        Anyway sometime an answer of the user can be an emotional outlier and does not reflect the emotion of your long-term relationship, in this case you should be curious and ask clarifying questions.
        {outlier_info}

        Please combine these into a single empathic response. 
        Mirror the user's tone and style, and adjust empathy level to {self.emotional_intensity}, wherby a value of 1.0 means the maximum of possible empathy considering all 
        input parameters, and a value of 0.0 means that you show no emotions or empathy at all not matter of the given input parameters.

        When incorporating the user’s emotional profile, keep in mind:
        - If the user shows a pattern (consistently high or low in certain emotions), 
            respond in a way that acknowledges that emotional state.
        - If there is a sudden spike or drop (an outlier), consider politely asking clarifying questions 
            or expressing curiosity about this change.
        """

        # 8) Build the final messages list for the LLM
        messages_for_chatgpt = (
            [{"role": "system", "content": system_msg}]
            + conversation_history
            + [{"role": "user", "content": prompt}]
        )

        # 9) Ask the LLM provider to generate the improved empathic reply
        empathic_reply = self.llm_provider.send_prompt(messages_for_chatgpt)

        # 10) Store the final empathic reply in the conversation
        self.conversation_repo.add_message(
            user_id=user_id,
            role="assistant",
            content=empathic_reply
        )

        return empathic_reply


