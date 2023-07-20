import os
import re
import uuid
import threading
import traceback

from flask import Flask, render_template, request, redirect, url_for, jsonify, session

from utils.engine import Engine

os.makedirs("info/")
os.makedirs("index/")
os.makedirs("static/uploads/")

app = Flask(__name__)
app.secret_key = "SUPER_SECRET_KEY"
engine = Engine()


@app.route("/clear")
def clear():
    try:
        if "uploads" not in session:
            session["uploads"] = []
        for file_id in session["uploads"]:
            upload_path = os.path.join("static/uploads/", file_id + ".pdf")
            info_path = os.path.join("info/", file_id + ".json")
            index_path = os.path.join("index/", file_id + ".npy")

            if os.path.isfile(upload_path):
                os.remove(upload_path)
            if os.path.isfile(info_path):
                os.remove(info_path)
            if os.path.isfile(index_path):
                os.remove(index_path)
        session.clear()
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for("error"))

    return redirect(url_for("index"))


@app.route("/")
def index():
    try:
        if "uploads" not in session:
            session["uploads"] = []

        states = []
        for i in range(len(session["uploads"])):
            file_id = session["uploads"][i]
            file_status = engine.get_status(file_id)
            states.append(file_status)

        return render_template("index.html", uploads=states)
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for("error"))


@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        file_id = str(uuid.uuid4())
        file.save(os.path.join("static/uploads/", file_id + ".pdf"))
        engine.save_file(file_id, file.filename)

        session_uploads = session["uploads"]
        session_uploads.append(file_id)
        session["uploads"] = session_uploads

        thread = threading.Thread(target=engine.index, args=(file_id,))
        thread.start()
        return redirect(url_for("index"))
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for("error"))


@app.route("/search")
def search():
    try:
        file_id = request.args.get("file_id")
        pdf_path = os.path.join("static/uploads/", file_id + ".pdf")
        return render_template("search.html", file_id=file_id, pdf_path=pdf_path)
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for("error"))


@app.route("/query")
def query():
    try:
        file_id = request.args.get("file_id")
        query = request.args.get("query")

        results = engine.retrieve(file_id, query)
        return jsonify({"results": results})
    except Exception as e:
        traceback.print_exc()
        return "Could not query", 400


@app.route("/insight", methods=["POST"])
def insight():
    try:
        data = request.get_json()

        file_id = data["file_id"]
        query = data["query"]
        doc_ids = data["documents"]

        prediction = engine.insight(file_id, query, doc_ids)
        # prediction = ''
        return jsonify({"prediction": prediction})
    except Exception as e:
        traceback.print_exc()
        return "Could not get insight", 400


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    email = request.form["email"]
    message = request.form["message"]

    if email == "":
        email = "EMPTY"
    message = re.sub(r"\s+", " ", message).strip()

    with open("feedback.txt", "a") as file:
        file.write(f"{email}, {message}\n")
    return redirect(url_for("index"))


@app.route("/error")
def error():
    return render_template("error.html")
