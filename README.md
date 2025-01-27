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

# References

Building Emotional Support Chatbots in the Era of LLMs: https://arxiv.org/pdf/2308.11584

DialogueLLM: Context and Emotion Knowledge-Tuned Large Language Models for Emotion Recognition in Conversations: https://arxiv.org/html/2310.11374v4

The Good, The Bad, and Why: Unveiling Emotions in Generative AI: https://arxiv.org/html/2312.11111v3#bib.bib39

## Citation

If you use Browser Use in your research or project, please cite:

```bibtex
@software{emotionsinai,
  author = {Gerrit Knippschild},
  title = {Emotionsin.ai - Empowering AI Agents with Emotions and Empathy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/emotionsinai}
}
```