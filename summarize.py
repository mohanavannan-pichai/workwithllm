# Hugging Face Model Demonstration with non Generative AI Language Models
# Loading Models with Authentication and Performing NLP Tasks
#By Mohanavannan Pichai

#This sample demonstrates:
#1.Create an simple GUI with TKinter Builtin Python Library
#2.Setting up Hugging Face authentication 
#3.Loading models for summarization
#4.Loading the large text from a text file
#5.Performing Summarization using 3 different models
#6.Show the results 

#Import all Required Libraries..
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import requests


import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForQuestionAnswering,
    pipeline
)
from huggingface_hub import login
import torch
import warnings
warnings.filterwarnings('ignore')
#Import ends here

class SummarizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Summarization Tests")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.access_token = tk.StringVar()
        self.is_logged_in = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Disconnected")

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        self.create_widgets()

    def create_widgets(self):
        # Top frame
        top_frame = ttk.Frame(self.main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        # Authentication section
        auth_frame = ttk.LabelFrame(top_frame, text="Hugging Face Authentication", padding="10")
        auth_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Status indicator
        status_frame = ttk.Frame(auth_frame)
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="red")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        # Access token input
        ttk.Label(auth_frame, text="Access Token:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.token_entry = ttk.Entry(auth_frame, textvariable=self.access_token, show="*", width=30)
        self.token_entry.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Login & Disconnect buttons
        self.login_btn = ttk.Button(auth_frame, text="Login to Hugging Face", command=self.handle_login)
        self.login_btn.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        self.disconnect_btn = ttk.Button(auth_frame, text="Disconnect", command=self.handle_disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Text input section

        text_frame = ttk.LabelFrame(top_frame, text="Text Input for Summarization", padding="10")
        text_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        #Read text from file
        try:
            input_text=""
            input_file=open("inputtext.txt","r")
            input_text=input_file.read()
        except FileNotFoundError:
            input_text="Enter the large text you want to summarize.."
        ttk.Label(text_frame, text="Enter the large text you want to summarize..").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.input_text = scrolledtext.ScrolledText(text_frame, height=10, width=50)
        self.input_text.insert(tk.INSERT,input_text)
        self.input_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Summarize button
        self.summarize_btn = ttk.Button(self.main_frame, text="Summarize", command=self.handle_summarize, state=tk.DISABLED)
        self.summarize_btn.grid(row=1, column=0, columnspan=2, pady=10)

        # Separator
        ttk.Separator(self.main_frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Results
        ttk.Label(self.main_frame, text="Summarization Results").grid(row=3, column=0, columnspan=2, pady=(0, 10))

        self.bart_result = self._create_result_box("BART Model", 4, 0)
        self.t5_result = self._create_result_box("T5 Model", 5, 0)
        self.mistral_result = self._create_result_box("Google Pegasus (Large)", 6, 0)

        # Separator
        ttk.Separator(self.main_frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Logs
        ttk.Label(self.main_frame, text="App Terminal").grid(row=8, column=0, columnspan=2, pady=(0, 10))

        self.app_logs = self._create_log_box("App Logs", 9, 0)


    def _create_result_box(self, label, row, col):
        frame = ttk.LabelFrame(self.main_frame, text=label, padding="5")
        frame.grid(row=row, column=col, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)
        box = scrolledtext.ScrolledText(frame, height=4, state=tk.DISABLED)
        box.grid(row=0, column=0, sticky=(tk.W, tk.E))
        return box

    def _update_result(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _create_log_box(self, label, row, col):
        frame = ttk.LabelFrame(self.main_frame, text=label, padding="5")
        frame.grid(row=row, column=col, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)
        box = scrolledtext.ScrolledText(frame, height=4, state=tk.DISABLED)
        box.grid(row=0, column=0, sticky=(tk.W, tk.E))
        return box

    def _update_log(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def handle_login(self):
        if not self.access_token.get().strip():
            messagebox.showwarning("Warning", "Please enter an access token")
            return

        # Simulate login
        self.login_btn.configure(text="Connecting...", state=tk.DISABLED)
        self.root.update()

        def login_process():
            #self.logging_process(f"Authenticated as: {user_info['name']}")
            try:
                login(token=self.access_token.get().strip())
                from huggingface_hub import HfApi
                api = HfApi()
                user_info = api.whoami()
                self.is_logged_in.set(True)
                self.root.after(0, self.login_success)
            except Exception as e:
                print("Not authenticated - will use public models only")
                self.root.after(0, self.login_failure)
                self.is_logged_in.set(False)
            time.sleep(1)
        threading.Thread(target=login_process, daemon=True).start()

    def login_success(self):
        self.is_logged_in.set(True)
        self.status_var.set("Connected")
        self.status_label.configure(foreground="green")
        self.login_btn.configure(text="Connected", state=tk.DISABLED)
        self.disconnect_btn.configure(state=tk.NORMAL)
        self.token_entry.configure(state=tk.DISABLED)
        self.summarize_btn.configure(state=tk.NORMAL)

    def login_failure(self):
        self.is_logged_in.set(False)
        self.status_var.set("Not Connected, check network connectivity or Enter Valid Access token")
        self.status_label.configure(foreground="red")
        self.login_btn.configure(text="Login",state=tk.NORMAL)
        self.disconnect_btn.configure(state=tk.DISABLED)
        self.token_entry.configure(state=tk.NORMAL)
        self.summarize_btn.configure(state=tk.NORMAL)

    def handle_disconnect(self):
        self.is_logged_in.set(False)
        self.status_var.set("Disconnected")
        self.status_label.configure(foreground="red")
        self.login_btn.configure(text="Login to Hugging Face", state=tk.NORMAL)
        self.disconnect_btn.configure(state=tk.DISABLED)
        self.token_entry.configure(state=tk.NORMAL)
        self.summarize_btn.configure(state=tk.DISABLED)
        self.access_token.set("")

        # Clear results
        for widget in [self.bart_result, self.t5_result, self.mistral_result, self.app_logs]:
            self._update_result(widget, "")
    def query_huggingface_t5(self,model,text):
        # Load Summarization Models
        print("Loading summarization models...")
        # Model T5 for summarization (good for instruction-following)
        summarizer_t5 = pipeline(
            "text2text-generation",
            model=model,  # Use t5-base or t5-large for better quality
            tokenizer="t5-small",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ T5 summarization model loaded successfully!")
        print(f"T5 model device: {summarizer_t5.device}")
        t5_input = "summarize: " + text
        t5_summary = summarizer_t5(t5_input, max_length=60, min_length=20, do_sample=False)
        return t5_summary[0]['generated_text']        

    def query_huggingface(self, model, text):
        if(self.is_logged_in):
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {self.access_token.get().strip()}"}
            payload = {"inputs": text}

            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and "summary_text" in data[0]:
                    return data[0]["summary_text"]
                elif isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                else:
                    return str(data)
            except Exception as e:
                return f"Error: {e}"
        else:
            return "Not Authenticated, check the network connection and access token"
    def handle_summarize(self):
        if not self.input_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "Please enter text to summarize")
            return
        if not self.is_logged_in.get():
            messagebox.showwarning("Warning", "Please login first")
            return

        self.summarize_btn.configure(text="Summarizing...", state=tk.DISABLED)
        input_text = self.input_text.get(1.0, tk.END).strip()

        def summarize_process():
            bart_summary = self.query_huggingface("facebook/bart-large-cnn", input_text)
            t5_summary = self.query_huggingface_t5("t5-small", input_text)
            mistral_summary = self.query_huggingface("google/pegasus-large", f"Summarize this: {input_text}")

            def update_ui():
                self._update_result(self.bart_result, bart_summary)
                self._update_result(self.t5_result, t5_summary)
                self._update_result(self.mistral_result, mistral_summary)
                self._update_log(self.app_logs, "Summarization completed successfully.")

                self.summarize_btn.configure(text="Summarize", state=tk.NORMAL)

            self.root.after(0, update_ui)

        threading.Thread(target=summarize_process, daemon=True).start()

# Update logs in a separate thread
"""
        def logging_process(log_text):
            def update_log():
                self._update_log(self.app_logs, log_text)
            self.root.after(0, update_log)
        threading.Thread(target=logging_process, daemon=True).start()
"""



if __name__ == "__main__":
    root = tk.Tk()
    app = SummarizationApp(root)
    root.mainloop()
