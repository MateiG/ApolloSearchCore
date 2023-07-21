import os
import re
import uuid
import threading
import traceback

from flask import Flask, render_template, request, redirect, url_for, jsonify

from utils.engine import Engine


app = Flask(__name__)
app.secret_key = "SUPER_SECRET_KEY"
engine = Engine()


@app.route("/clear")
def clear():
    try:
        for f in os.listdir("info/"):
            os.remove("info/" + f)
        for f in os.listdir("index/"):
            os.remove("index/" + f)
        for f in os.listdir("static/uploads/"):
            os.remove("static/uploads/" + f)
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for("error"))

    return redirect(url_for("index"))


@app.route("/")
def index():
    try:
        states = []
        for f in os.listdir("info/"):
            file_id = f.replace(".json", "")
            file_status = engine.get_status(file_id)
            states.append(file_status)
        return render_template("index.html", uploads=states)
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for("error"))


@app.route("/upload", methods=["POST"])
def upload():
    try:
        for file in request.files.getlist("file"):
            file_id = str(uuid.uuid4())
            file.save(os.path.join("static/uploads/", file_id + ".pdf"))
            engine.save_file(file_id, file.filename)

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


if __name__ == "__main__":
    os.makedirs("info/", exist_ok=True)
    os.makedirs("index/", exist_ok=True)
    os.makedirs("static/uploads/", exist_ok=True)

    app.run(host="0.0.0.0", port="5000", debug=True)
