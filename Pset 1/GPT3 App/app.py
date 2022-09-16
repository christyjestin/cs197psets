import os

import openai
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods = ("GET", "POST"))
def index():
    if request.method == "POST":
        task = request.form["task"]
        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = generate_prompt(task),
            temperature = 0.1,
            max_tokens = 640
        )
        return redirect(url_for("index", result = response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result = result)


def generate_prompt(task):
    return f"You need to explain how to {task.lower()} to a robot. You assume that \
            the robot knows nothing and give it very simple step by step instructions:"
