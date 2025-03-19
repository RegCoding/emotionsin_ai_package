# main.py

import logging
import os
import threading
import time
import tkinter as tk
from tkinter import font, scrolledtext, messagebox
from dotenv import load_dotenv

from emotionsinai import OpenAIProvider
from emotionsinai import OllamaProvider
from emotionsinai import EmotionServices

# Configure logging
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ChatApp:
    def __init__(self, root, emotion_service, user_id):
        self.root = root
        self.emotion_service = emotion_service
        self.user_id = user_id

        # Configure the root window
        self.root.title("Conversational Assistant")
        self.root.configure(bg="white")

        # Define Roboto font (make sure Roboto is installed; if not, Tkinter will fall back to a default font)
        self.roboto_font = font.Font(family="Roboto", size=12)

        # Create a scrolled text widget for conversation output
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=self.roboto_font, bg="white", state=tk.DISABLED)
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create a frame to hold the entry and button
        self.entry_frame = tk.Frame(root, bg="white")
        self.entry_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Entry widget for user input
        self.user_entry = tk.Entry(self.entry_frame, font=self.roboto_font)
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_entry.bind("<Return>", self.process_input)

        # Send button
        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.process_input, font=self.roboto_font)
        self.send_button.pack(side=tk.RIGHT)

        # Start polling for new responses from emotion_service
        self.poll_responses()

        # Welcome message
        self.append_text("System: Welcome to the conversational assistant! Type 'exit' to end the conversation.\n")

    def append_text(self, message):
        """Append a message to the text widget."""
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert(tk.END, message + "\n")
        self.text_area.configure(state=tk.DISABLED)
        self.text_area.see(tk.END)

    def process_input(self, event=None):
        """Get user input, display it and send it to the emotion service."""
        user_prompt = self.user_entry.get().strip()
        if not user_prompt:
            return

        # Display user input in the text widget
        self.append_text(f"You: {user_prompt}")
        logging.info(f"User prompt: {user_prompt}")

        if user_prompt.lower() == "exit":
            self.append_text("System: Goodbye! Thank you for the conversation.")
            logging.info("Conversation ended by user.")
            self.root.quit()
            return

        # Send the user prompt to the emotion service
        try:
            self.emotion_service.add_input(self.user_id, user_prompt)
        except Exception as e:
            logging.error(f"An error occurred while sending input: {e}")
            messagebox.showerror("Error", "An error occurred. Please check the log file for details.")

        # Clear the entry widget
        self.user_entry.delete(0, tk.END)

    def poll_responses(self):
        """Periodically check for new responses from the emotion service."""
        try:
            new_response = self.emotion_service.get_new_response()
            if new_response is not None:
                self.append_text(f"Assistant (async): {new_response}")
                logging.info(f"Async Emotionsin.ai reply: {new_response}")
        except Exception as e:
            logging.error(f"Error polling responses: {e}")
        # Schedule this method to be called again after 1 second
        self.root.after(1000, self.poll_responses)

def main():
    logging.info("Starting the application...")

    # Initialize providers
    thinking_llm = OpenAIProvider(model_name="gpt-4", temperature=0.7, openai_key=OPENAI_API_KEY)
    reflection_llm = OllamaProvider(model_name="llama3.1")
    emotion_service = EmotionServices(thinking=thinking_llm, reflecting=reflection_llm, resource_file_path="resources.json")
    user_id = "user321"  # In practice, this might be a session ID or database ID

    # Create the GUI window
    root = tk.Tk()
    app = ChatApp(root, emotion_service, user_id)
    
    # Start the Tkinter event loop
    try:
        root.mainloop()
    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}")
        print("An error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()
