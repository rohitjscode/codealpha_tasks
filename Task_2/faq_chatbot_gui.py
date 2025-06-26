import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# FAQ dataset
faq_data = {
    "What is your return policy?": "Our return policy allows returns within 30 days of purchase.",
    "How do I track my order?": "You can track your order using the tracking link sent to your email.",
    "Do you offer international shipping?": "Yes, we ship to most countries worldwide.",
    "What payment methods are accepted?": "We accept credit cards, debit cards, and PayPal.",
    "How do I cancel my order?": "You can cancel your order from your account before it is shipped."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# Preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# Match answer
def get_answer(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    return answers[index]

# GUI Functionality
def send_message(event=None):
    user_msg = user_input.get()
    if user_msg.strip() == "":
        return
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_msg}\n")
    response = get_answer(user_msg)
    chat_window.insert(tk.END, f"Bot: {response}\n\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.see(tk.END)
    user_input.delete(0, tk.END)

# GUI Setup
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("500x550")

# Chat area
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_window.config(state=tk.DISABLED)

# Input area
frame = tk.Frame(root)
frame.pack(padx=10, pady=5, fill=tk.X)

user_input = tk.Entry(frame, font=("Arial", 14))
user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
user_input.bind("<Return>", send_message)  # Press Enter to send

send_btn = tk.Button(frame, text="Send", font=("Arial", 12), command=send_message)
send_btn.pack(side=tk.RIGHT)

# Start GUI loop
root.mainloop()
