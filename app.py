import os
import re
import uuid
import threading
import traceback

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename

from mixpanel import Mixpanel

from utils.engine import Engine

app = Flask(__name__)
app.secret_key = 'SUPER_SECRET_KEY'
mp = Mixpanel(os.getenv('MIXPANEL_KEY', '091f7f4f16d98b2155901f950b488c1b'))
engine = Engine()


@app.route('/clear')
def clear():
    try:
        mp.track(request.remote_addr, 'clear')

        if ('uploads' not in session):
            session['uploads'] = []
        for upload in session['uploads']:            
            upload_path = os.path.join('static/uploads/', upload['id'] + '.pdf')
            index_path = os.path.join('index/', upload['id'] + '.json')

            if (os.path.isfile(upload_path)):
                os.remove(upload_path)
            if (os.path.isfile(index_path)):
                os.remove(index_path)
        session.clear()
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for('error'))
    
    return redirect(url_for('index'))


@app.route('/')
def index():
    try:
        mp.track(request.remote_addr, 'visit')

        if ('uploads' not in session):
            session['uploads'] = []

        for i in range(len(session['uploads'])):
            upload = session['uploads'][i]
            if (os.path.exists('index/' + upload['id'] + '.json')):
                upload['status'] = 'Ready'
            else:
                upload['status'] = 'Processing'

        return render_template('index.html', uploads=session['uploads'])
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for('error'))


@app.route('/upload', methods=['POST'])
def upload():
    try:
        mp.track(request.remote_addr, 'upload')

        file = request.files['file']
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        save_path = os.path.join('static/uploads/', file_id + '.pdf')
        file.save(save_path)

        session_uploads = session['uploads']
        session_uploads.append({
            'id': file_id,
            'name': filename,
            'status': 'Processing'
        })
        session['uploads'] = session_uploads

        # num_pages = engine.get_num_pages(file_id)
        # if (num_pages <= 15):
        #     engine.index(file_id, filename, is_short=True)
        # else:
            # start multithreading
        thread = threading.Thread(target=engine.index, args=(file_id, filename, False))
        thread.start()
        return redirect(url_for('index'))
        # return redirect(url_for('search', file_id=file_id))
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for('error'))


@app.route('/search')
def search():
    try:
        mp.track(request.remote_addr, 'search')

        file_id = request.args.get('file_id')
        index = engine.read(file_id=file_id)

        pdf_path = os.path.join('static/uploads/', file_id + '.pdf')
        filename = index['name']
        return render_template('search.html', file_id=file_id, filename=filename, pdf_path=pdf_path)
    except Exception as e:
        traceback.print_exc()
        return redirect(url_for('error'))


@app.route('/query')
def query():
    try:
        mp.track(request.remote_addr, 'query')

        file_id = request.args.get('file_id')
        query = request.args.get('query')

        results = engine.retrieve(file_id, query)
        return jsonify({'results': results})
    except Exception as e:
        traceback.print_exc()
        return 'Could not query', 400


@app.route('/insight', methods=['POST'])
def insight():
    try:
        mp.track(request.remote_addr, 'insight')

        data = request.get_json()

        file_id = data['file_id']
        query = data['query']
        doc_ids = data['documents']

        prediction = engine.insight(file_id, query, doc_ids)
        return jsonify({'prediction': prediction})
    except Exception as e:
        traceback.print_exc()
        return 'Could not get insight', 400


@app.route('/feedback')
def feedback():
    return render_template('feedback.html')


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    mp.track(request.remote_addr, 'feedback')

    email = request.form['email']
    message = request.form['message']

    if (email == ''):
        email = 'EMPTY'
    if (email == ''):
        message = ''
    message = re.sub(r'\s+', ' ', message).strip()

    with open('feedback.txt', 'a') as file:
        file.write(f"{email}, {message}\n")
    return redirect(url_for('index'))


@app.route('/error')
def error():
    return render_template('error.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
