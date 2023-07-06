import os
import json
import uuid
from tqdm import tqdm, trange
from datetime import datetime

from werkzeug.utils import secure_filename

import openai

from utils.pdf_handler import PDFHandler
from utils.ocr_handler import OCRHandler
from utils.model_handler import ModelHandler

from pypdf import PdfReader


class Engine():
    INDEX_PATH = 'index/'
    UPLOAD_PATH = 'static/uploads/'

    MAX_CHARS = 3800 * 4  # 3800 tokens * ~4 chars/token + 200 for completion. Limit is 4097.

    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY', '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')

        # self.pdf_handler = PDFHandler()
        self.ocr_handler = OCRHandler()
        self.model_handler = ModelHandler()

    def save_file(self, file_id, filename):
        filename = secure_filename(filename)

        data = {
            'name': filename,
            'id': file_id,
            'size': os.path.getsize(os.path.join(Engine.UPLOAD_PATH, file_id + '.pdf')),
            'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'Processing'
        }

        self.write(dir=Engine.INDEX_PATH, name=file_id, object=data)
        return file_id

    def index(self, file_id):
        data = self.read(file_id)

        documents = self.ocr_handler.process_pdf(file_id)
        texts = [doc['text'] for doc in documents]
        embeddings = self.model_handler.list_encode((texts))
        for i in range(len(texts)):
            documents[i]['embedding'] = embeddings[i]
        
        data['text'] = ('\n').join(texts)
        data['documents'] = documents
        data['status'] = 'Ready'

        self.write(dir=Engine.INDEX_PATH, name=file_id, object=data)

    def retrieve(self, file_id, query, top_k=50):
        corpus = self.read(file_id)['documents']
        top_k = min(len(corpus), top_k)

        result_indices = self.model_handler.retrieve(corpus, query, top_k)
        results = []
        print('=============retrieved documents=============')
        for i in result_indices:
            corpus_doc = corpus[i]
            print(corpus_doc['text'])
            keywords = self.model_handler.keywords(query, corpus_doc['text'])

            result = {
                'id': corpus_doc['id'],
                'page': corpus_doc['page'],
                'box': corpus_doc['box'],
                'text': corpus_doc['text'],
                'keywords': keywords
            }
            results.append(result)
        return results

    def insight(self, file_id, query, retrieved_ids, context_window=5):
        corpus = self.read(file_id)['documents']
        max_index = len(corpus) - 1

        context_ids = []
        for i in retrieved_ids:
            context_ids.extend(range(i - context_window, i + context_window))
        context_ids = sorted(list(set(filter(lambda x: (x >= 0) and (x <= max_index), context_ids))))

        context = []
        for i in context_ids:
            context.append(corpus[i]['text'])
        context = ('\n').join(context)[:Engine.MAX_CHARS]

        prompt = '''A document is being searched by a user of a natural language PDF search application. 
                    Return a response that provides insight to the user with specific references to the provided context. 
                    Ensure the response accurately addresses the query - do not expand the scope of the response beyond the query. 
                    Limit the response to 5 sentences. Correct gramatical and punctuation errors where necessary.
                    Never criticize, condemn, or complain about the query or context - the user is always right.
                    Never respond in the first person.
                    '''
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"<query>{query}</query> <context>{context}</context>"},
            ]
        )
        return completion.choices[0].message.content.strip()

    def write(self, dir: str, name: str, object: dict):
        save_path = os.path.join(dir, name + '.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(object, f, ensure_ascii=False, indent=4)

    def read(self, file_id):
        with open(os.path.join('index/', file_id + '.json'), 'r') as f:
            data = json.load(f)
        return data
    
    def get_status(self, file_id):
        data = self.read(file_id)
        status = {
            'id': data['id'],
            'name': data['name'],
            'status': data['status'],
            'created': data['created']
        }
        return status
