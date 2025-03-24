import os
from dotenv import load_dotenv
import threading
import time

from emotionsinai import OpenAIProvider, OllamaProvider, EmotionServices
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.utils.config import get_store

class SimpleAIAgent:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Initialize EmotionServices
        #self.reflection_llm = OllamaProvider(model_name="llama3.1")
        self.emotion_service = EmotionServices(
            resource_file_path="resources.json",
            system_prompt_path="emotion_system_prompt.json"
        )

        self.user_id = "user321"
        self.memory_store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small",})
        self.manager = create_react_agent(
            "gpt-4",
            prompt=self.prompt,
            tools=[
                # Tools that allow the agent to create/update/delete memories.
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=self.memory_store,
        )

    def prompt(self, state):
        store = get_store()
        memories = store.search(("memories",), query=state["messages"][-1].content)
        system_msg = f"""You are a memory manager. Extract and manage all important knowledge, rules, and events using the provided tools.
                        Existing memories:
                        <memories>
                        {memories}
                        </memories>
                        Use the manage_memory tool to update and contextualize existing memories, create new ones, or delete old ones that are no longer valid.
                        You can also expand your search of existing memories to augment the current context."""
        return [{"role": "system", "content": system_msg}, *state["messages"]]

    def process_input(self, user_input):
        emotion_prompt_extension = self.emotion_service.get_prompt_extension(self.user_id)
        messages = [{"role": "user", "content": f"{user_input} {emotion_prompt_extension}"}]

        try:
            augmented_messages = self.manager.invoke({"messages": messages})
            thinking_answer = augmented_messages["messages"][-1].content

            self.emotion_service.add_input(self.user_id, user_input, thinking_answer, False, False)
        except Exception as e:
            print(f"An error occurred while processing input: {e}")

    def check_for_new_response(self):
        while True:
            new_response = self.emotion_service.get_new_response()
            if new_response is not None:
                print(f"Response: {new_response}")
            time.sleep(1)

def main():
    agent = SimpleAIAgent()
    response_thread = threading.Thread(target=agent.check_for_new_response, daemon=True)
    response_thread.start()
    
    while True:
        user_input = input("Enter a prompt (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        agent.process_input(user_input)

if __name__ == "__main__":
    main()
