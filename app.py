import os
import uuid
import threading

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename

from utils.engine import Engine

app = Flask(__name__)
app.secret_key = 'SUPER_SECRET_KEY'
engine = Engine()

@app.route('/clear')
def clear():
    files = os.listdir('index/')
    for file_name in files:
        file_path = os.path.join('index/', file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    files = os.listdir('static/uploads/')
    for file_name in files:
        file_path = os.path.join('static/uploads/', file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    session.clear()
    return 'OK'



@app.route('/')
def index():
    if ('uploads' not in session):
        session['uploads'] = []
    
    for i in range(len(session['uploads'])):
        upload = session['uploads'][i]
        if (os.path.exists('index/' + upload['id'] + '.json')):
            upload['status'] = 'Ready'
        else:
            upload['status'] = 'Processing'

    return render_template('index.html', uploads=session['uploads'])

@app.route('/upload', methods=['POST'])
def upload():
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

    num_pages = engine.get_num_pages(file_id)
    if (num_pages <= 15):
        engine.index(file_id, filename, is_short=True)
    else:
        # start multithreading
        thread = threading.Thread(target=engine.index, args=(file_id, filename, False))
        thread.start()
        return redirect(url_for('index'))
    return redirect(url_for('search', file_id=file_id))

@app.route('/search')
def search():
    file_id = request.args.get('file_id')
    index = engine.read(file_id=file_id)
    
    pdf_path = os.path.join('static/uploads/', file_id + '.pdf')
    filename = index['name']
    return render_template('search.html', file_id=file_id, filename=filename, pdf_path=pdf_path, adobe_key=os.getenv('ADOBE_KEY', '02fc6f86eea34c389266060a4b48d938'))

@app.route('/query')
def query():
    file_id = request.args.get('file_id')
    query = request.args.get('query')

    results = engine.retrieve(file_id, query)
    return jsonify({'results': results})

@app.route('/insight', methods=['POST'])
def insight():
    data = request.get_json()

    file_id = data['file_id']
    query = data['query']
    doc_ids = data['documents']

    prediction = engine.insight(file_id, query, doc_ids)
    return jsonify({'prediction': prediction})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
