import os
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, jsonify, render_template, request

from nitbot.index import chat_input


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/callbot/")
def callbot_api():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("userMessage") or "").strip()

    if not user_input:
        return jsonify({"botText": ""})

    response = chat_input(user_input)
    return jsonify({"botText": response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
