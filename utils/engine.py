import os
import json
import uuid
from tqdm import tqdm, trange
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

import openai

from utils.pdf_handler import PDFHandler


class Engine():
    encoder_save_path='models/all-MiniLM-L6-v2'
    cross_save_path='models/ms-marco-MiniLM-L-2-v2'
    qa_save_path='models/tinyroberta-squad2'

    INDEX_PATH='index/'

    def __init__(self):
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(Engine.encoder_save_path, )
        self.encoder_model = AutoModel.from_pretrained(Engine.encoder_save_path)
        self.encoder_model.eval()

        self.cross_tokenizer = AutoTokenizer.from_pretrained(Engine.cross_save_path)
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(Engine.cross_save_path)
        self.cross_model.eval()

        self.qa_tokenizer = AutoTokenizer.from_pretrained(Engine.qa_save_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(Engine.qa_save_path)
        self.qa = pipeline('question-answering', model=self.qa_model, tokenizer=self.qa_tokenizer)

        self.pdf_handler = PDFHandler()
        self.cos_sim = CosineSimilarity()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, documents):
        encoded_input = self.encoder_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.encoder_model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy().tolist()
    
    def highlight(self, query, context, chars: int = 50):
        res = self.qa({'question': query, 'context': context})
        start = max(0, res['start'] - chars)
        end = min(res['end'] + chars, len(context))

        return start, end # res['answer']

    def index(self, file_id, filename, is_short=True):
        print('starting indexing')
        if (is_short):
            documents = self.pdf_handler.online_process(file_id)
        else:
            documents = self.pdf_handler.offline_process(file_id)
        print('finished processing, generating embeddings')
        texts = [doc['text'] for doc in documents]

        embeddings = self.encode(texts)
        for i in range(len(texts)):
            documents[i]['embedding'] = embeddings[i]

        text = ('\n').join(texts)
        index = {'name': filename, 'text': text, 'documents': documents}

        self.write(dir=Engine.INDEX_PATH, name=file_id, object=index)
        return index
    
    def retrieve(self, file_id, query, top_k=15):
        corpus = self.read(file_id)['documents']
        texts = [doc['text'] for doc in corpus]
        top_k = min(len(corpus), top_k)

        corpus_emb = torch.tensor([doc['embedding'] for doc in corpus])
        query_emb = torch.tensor(self.encode([query]))

        similarities = self.cos_sim(query_emb, corpus_emb)
        values, indices = torch.topk(similarities, k=top_k)
        reranked_indices = self.rerank(query, texts, indices.tolist())
        
        results = []
        for i in reranked_indices:
            corpus_doc = corpus[i]
            h_start, h_end = self.highlight(query, corpus_doc['text'])

            result = {
                'id': corpus_doc['id'],
                'text': corpus_doc['text'],
                'h_start': h_start,
                'h_end': h_end
            }
            results.append(result)
        return results
    
    def rerank(self, query, texts, indices):
        num_indices = len(indices)
        top_texts = [texts[i] for i in indices]
        features = self.cross_tokenizer([query] * num_indices, top_texts, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            scores = self.cross_model(**features).logits
            scores = torch.flatten(scores)

        cross_values, cross_indices = torch.topk(scores, k=num_indices)
        return [indices[i] for i in cross_indices.tolist()]

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
        context = ('\n').join(context)

        prompt = f'''Answer this prompt: {query}, with specific references to the following context:\n
            {context}\n
            Limit to 5 short sentences that are easy to understand. 
            Correct gramatical and punctuation errors where necessary.'''
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
