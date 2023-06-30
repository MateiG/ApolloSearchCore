import os
import json
import uuid
from tqdm import tqdm, trange
from datetime import datetime

import openai
from nltk.tokenize import word_tokenize

from utils.pdf_handler import PDFHandler
from utils.model_handler import ModelHandler

from pypdf import PdfReader


class Engine():
    INDEX_PATH = 'index/'
    UPLOAD_PATH = 'static/uploads/'

    MAX_CHARS = 3800 * 4  # 3800 tokens * ~4 chars/token + 200 for completion. Limit is 4097.

    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY', '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')

        self.pdf_handler = PDFHandler()
        self.model_handler = ModelHandler()

    def index(self, file_id, filename, is_short=True):
        if (is_short):
            documents = self.pdf_handler.online_process(file_id)
        else:
            documents = self.pdf_handler.offline_process(file_id)

        texts = [doc['text'] for doc in documents]
        embeddings = self.model_handler.encode(texts)

        for i in range(len(texts)):
            documents[i]['embedding'] = embeddings[i]

        text = ('\n').join(texts)
        date = datetime.now()
        index = {'name': filename, 'date': date.strftime("%Y-%m-%d %H:%M:%S"), 'text': text, 'documents': documents}

        self.write(dir=Engine.INDEX_PATH, name=file_id, object=index)
        return index

    def retrieve(self, file_id, query, top_k=50):
        corpus = self.read(file_id)['documents']
        top_k = min(len(corpus), top_k)

        result_indices = self.model_handler.retrieve(corpus, query, top_k)
        results = []
        for i in result_indices:
            corpus_doc = corpus[i]
            # h_start, h_end = self.model_handler.highlight(query, corpus_doc['text'])
            h_start, h_end = 0, 0

            result = {
                'id': corpus_doc['id'],
                'page': corpus_doc['page'],
                'box': corpus_doc['box'],
                'text': corpus_doc['text'],
                'h_start': h_start,
                'h_end': h_end
            }
            results.append(result)
        return results

    def insight(self, file_id, query, retrieved_ids, context_window=3):
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

        prompt = f'''A document is being searched using this query: {query}.\n
                    The following paragraphs from this document have been identified as containing relevant context: {context}.\n
                    Return a response that provides insight to the user with specific references to the aforementioned context.
                    Ensure the response accurately addresses the query - do not expand the scope of the response beyond the query.
                    Limit the response to 5 sentences (shorter is better). Correct gramatical and punctuation errors where necessary.
                    '''
        completion = openai.Completion.create(model='text-davinci-003', prompt=prompt, max_tokens=200, temperature=0.3)

        return completion.choices[0].text.strip()

    def write(self, dir: str, name: str, object: dict):
        save_path = os.path.join(dir, name + '.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(object, f, ensure_ascii=False, indent=4)

    def read(self, file_id):
        with open(os.path.join('index/', file_id + '.json'), 'r') as f:
            data = json.load(f)
        return data

    def get_num_pages(self, file_id):
        reader = PdfReader(os.path.join(Engine.UPLOAD_PATH, file_id + '.pdf'))
        return len(reader.pages)
