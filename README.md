# Quick Start

Install with pip:

```bash
pip install emotionsinai
```

# Setup EmotionsinAI

```python
from emotionsinai import OpenAIProvider
from emotionsinai import ConversationRepository
from emotionsinai import EmotionServices

def main():

    # Decide which LLM provider you prefer. You can simply add your own provider here by using a derivative of BaseLLM.
    # For this example we use OpenAIProvider with GPT-4.
    provider = OpenAIProvider(model_name="gpt-4", temperature=0.7, openai_key=OPENAI_API_KEY)
    # Initiate the EmotionService with the chosen conversation_repo and llm provider
    emotion_service = EmotionServices(conversation_repo, llm_provider=provider, resource_file_path="resources.json")

```