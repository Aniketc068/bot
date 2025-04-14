from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

import json
import difflib
import re
import os
import random
import uuid
import torch

app = Flask(__name__)
app.secret_key = "your_secret_key"
CORS(app)


greeting_responses = {
    "hi": "Hello! 😊",
    "hello": "Hi there! 👋",
    "hey": "Hey! How can I help you today?",
    "how are you": "I'm doing great, thanks! What about you?",
    "good morning": "Good morning! 🌞",
    "good evening": "Good evening! 🌆"
}

def is_greeting(text):
    text = clean_text(text)
    return greeting_responses.get(text)


# -------------------- Utils --------------------
def is_valid_phone(phone):
    return re.match(r'^(\+91)?[6-9]\d{9}$', phone)

def is_valid_email(email):
    return '@' in email

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def extract_keywords(text):
    return set(re.findall(r'\b\w+\b', clean_text(text)))


def format_answer(answer):
    # Convert URLs to clickable links
    answer = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank">\1</a>', answer)
    # Replace bullet points with line breaks
    answer = answer.replace("•", "\n•")
    return answer

# -------------------- Load Q&A --------------------
QA_PATH = "qa_data.json"

with open(QA_PATH, "r", encoding="utf-8") as f:
    qa_map = json.load(f)

# Load questions and corresponding answers
question_texts = []
answer_map = []

for group in qa_map.values():
    for q in group["questions"]:
        question_texts.append(q)
        answer_map.append(group["answer"])

# -------------------- Load Model --------------------
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(question_texts, convert_to_tensor=True)
print("Model loaded and question embeddings ready ✅")

# -------------------- Load/Save User Data --------------------
USER_DATA_PATH = "user_data.json"

def load_user_data():
    if not os.path.exists(USER_DATA_PATH) or os.stat(USER_DATA_PATH).st_size == 0:
        return {}
    with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user_data(data):
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

user_data = load_user_data()

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

# @app.route("/ask", methods=["POST"])
# def ask():
#     global user_data

#     data = request.get_json()
#     original_input = data.get("question", "").strip()
#     session_id = session.get("user_id")

#     # New user session
#     if not session_id:
#         session_id = str(uuid.uuid4())
#         session["user_id"] = session_id

#     if session_id not in user_data:
#         user_data[session_id] = {
#             "name": None,
#             "phone": None,
#             "email": None,
#             "chat_history": []
#         }

#     user = user_data[session_id]
#     user["chat_history"].append({"user": original_input})

#     # ---------------- Collect Info Phase ----------------
#     if not user["name"]:
#         user["name"] = original_input
#         save_user_data(user_data)
#         return jsonify({"answer": f"Thanks {original_input}! Please share your phone number."})

#     elif not user["phone"]:
#         if not is_valid_phone(original_input):
#             return jsonify({"answer": "Invalid phone number. Please enter a valid number like +919876543210 or 9876543210."})
#         user["phone"] = original_input
#         return jsonify({"answer": f"{user['name']}, please share your email address."})

#     elif not user["email"]:
#         if not is_valid_email(original_input):
#             return jsonify({"answer": "Invalid email. Please enter a valid email address containing '@'."})
#         user["email"] = original_input
#         save_user_data(user_data)
#         return jsonify({"answer": f"Awesome {user['name']}! We can now start the questions. Ask me anything 😊"})
    

#     # Check for greetings
#     greet_reply = is_greeting(original_input)
#     if greet_reply:
#         user["chat_history"].append({"bot": greet_reply})
#         save_user_data(user_data)
#         return jsonify({"answer": greet_reply})

#     # ---------------- Translate Input ----------------
#     try:
#         translated_input = GoogleTranslator(source='auto', target='en').translate(original_input)
#     except Exception:
#         translated_input = original_input

#     print(f"Original: {original_input} → Translated: {translated_input}")

#     # ---------------- Embedding & Similarity ----------------
#     input_embedding = model.encode(translated_input, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0]

#     top_score_idx = torch.argmax(cosine_scores).item()
#     top_score = cosine_scores[top_score_idx].item()

#     print(f"Best Match Score: {top_score:.2f}")

#     if top_score >= 0.75:
#         matched_answer = answer_map[top_score_idx]
#         if isinstance(matched_answer, list):
#             matched_answer = random.choice(matched_answer)
#         user["unanswered_count"] = 0
#     else:
#         matched_answer = "Sorry, I don't know the answer to that yet. 😅"
#         user.setdefault("unanswered_count", 0)
#         user["unanswered_count"] += 1

#     # ---------------- Handle Fallback ----------------
#     if user["unanswered_count"] >= 3:
#         support_msg = "Please contact our technical support team at 011-61400000 📞"
#         matched_answer += f"\n\n{support_msg}"
#         user["unanswered_count"] = 0

#     user["chat_history"].append({"bot": matched_answer})
#     save_user_data(user_data)

#     return jsonify({"answer": matched_answer})



@app.route("/ask", methods=["POST"])
def ask():
    global user_data, question_embeddings

    data = request.get_json()
    original_input = data.get("question", "").strip()
    session_id = session.get("user_id")

    if not session_id:
        session_id = str(uuid.uuid4())
        session["user_id"] = session_id

    if session_id not in user_data:
        user_data[session_id] = {
            "name": None,
            "phone": None,
            "email": None,
            "chat_history": []
        }

    user = user_data[session_id]
    user["chat_history"].append({"user": original_input})

    # Collect Info Phase
    if not user["name"]:
        user["name"] = original_input
        save_user_data(user_data)
        return jsonify({"answer": f"Thanks {original_input}! Please share your phone number."})

    elif not user["phone"]:
        if not is_valid_phone(original_input):
            return jsonify({"answer": "Invalid phone number. Please enter a valid number like +919876543210 or 9876543210."})
        user["phone"] = original_input
        return jsonify({"answer": f"{user['name']}, please share your email address."})

    elif not user["email"]:
        if not is_valid_email(original_input):
            return jsonify({"answer": "Invalid email. Please enter a valid email address containing '@'."})
        user["email"] = original_input
        save_user_data(user_data)
        return jsonify({"answer": f"Awesome {user['name']}! We can now start the questions. Ask me anything 😊"})

    # Check for greetings
    greet_reply = is_greeting(original_input)
    if greet_reply:
        user["chat_history"].append({"bot": greet_reply})
        save_user_data(user_data)
        return jsonify({"answer": greet_reply})

    if user.get("pending_questions"):
        new_answer = original_input.strip()

        if clean_text(new_answer) in ["i dont know", "dont know", "no idea", "not sure", "idk", "no"]:
            user["pending_questions"].pop(0)
            save_user_data(user_data)
            return jsonify({
                "answer": "No worries! 😊 Please contact our technical support at 📞 011-61400000 for help with this query."
            })

        # Load current data
        with open(QA_PATH, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        # Determine next question key
        current_max = max([int(k.replace("Question", "")) for k in qa_data.keys() if k.startswith("Question")], default=0)

        # Pop the first pending question
        new_question = user["pending_questions"].pop(0)
        new_answer = format_answer(new_answer)

        # Save it
        new_key = f"Question{current_max + 1:03d}"
        qa_data[new_key] = {
            "questions": [new_question],
            "answer": new_answer
        }

        with open(QA_PATH, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)

        # Update in-memory
        question_texts.append(new_question)
        answer_map.append(new_answer)
        new_embedding = model.encode([new_question], convert_to_tensor=True)
        global question_embeddings
        question_embeddings = torch.cat([question_embeddings, new_embedding], dim=0)

        save_user_data(user_data)

        return jsonify({
            "answer": "Thanks! I've saved your answer and learned something new. 🙌"
        })


    # Translate input
    try:
        translated_input = GoogleTranslator(source='auto', target='en').translate(original_input)
    except Exception:
        translated_input = original_input

    print(f"Original: {original_input} → Translated: {translated_input}")

    input_embedding = model.encode(translated_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0]

    top_score_idx = torch.argmax(cosine_scores).item()
    top_score = cosine_scores[top_score_idx].item()

    print(f"Best Match Score: {top_score:.2f}")

    if top_score >= 0.75:
        matched_answer = answer_map[top_score_idx]
        if isinstance(matched_answer, list):
            matched_answer = random.choice(matched_answer)
        user["unanswered_count"] = 0
    else:
        matched_answer = "Sorry, I don't know the answer to that yet. 😅\nCan you please tell me the correct answer in a single, complete message?"
        user.setdefault("unanswered_count", 0)
        user["unanswered_count"] += 1
        user.setdefault("pending_questions", [])
        user["pending_questions"].append(original_input)

    if user.get("unanswered_count", 0) >= 3:
        matched_answer += "\n\nPlease contact our technical support at 📞 011-61400000"
        user["unanswered_count"] = 0

    user["chat_history"].append({"bot": matched_answer})
    save_user_data(user_data)

    return jsonify({"answer": matched_answer})


@app.route("/history", methods=["GET"])
def history():
    session_id = session.get("user_id")
    if not session_id or session_id not in user_data:
        return jsonify({"history": []})
    return jsonify({"history": user_data[session_id].get("chat_history", [])})

# -------------------- Run Server --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

