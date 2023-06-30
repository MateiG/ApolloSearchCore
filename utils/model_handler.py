import os
import json
import uuid
from tqdm import tqdm, trange
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

class ModelHandler():
    encoder_save_path = 'models/splade_v2_max' # 'models/all-mpnet-base-v2'
    cross_save_path = 'models/ms-marco-MiniLM-L-12-v2' # 'models/ms-marco-MiniLM-L-2-v2'
    qa_save_path = 'models/tinyroberta-squad2'

    INDEX_PATH='index/'

    def __init__(self):
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.encoder_save_path)
        self.encoder_model = AutoModelForMaskedLM.from_pretrained(ModelHandler.encoder_save_path)
        self.encoder_model.eval()

        self.cross_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.cross_save_path)
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(ModelHandler.cross_save_path)
        self.cross_model.eval()

        self.qa_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.qa_save_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(ModelHandler.qa_save_path)
        self.qa = pipeline('question-answering', model=self.qa_model, tokenizer=self.qa_tokenizer)
        
    def encode(self, texts, chunk_size=50):
        print(f'encoding {len(texts)} texts')
        embeddings = torch.empty(0)
        with torch.no_grad():
            for i in trange(0, len(texts), chunk_size):
                chunk = texts[i:i+chunk_size]
                tokens = self.encoder_tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
                output = self.encoder_model(**tokens)
                vecs = torch.sum(torch.log(1 + torch.relu(output['logits'])) * tokens['attention_mask'].unsqueeze(-1), dim=1)
                embeddings = torch.cat((embeddings, vecs))
        print('finished encoding texts')
        return embeddings.cpu().numpy().tolist()
    
    def highlight(self, query, context, chars: int = 0):
        res = self.qa({'question': query, 'context': context})
        start = max(0, res['start'] - chars)
        end = min(res['end'] + chars, len(context))

        return start, end # res['answer']
    
    def retrieve(self, corpus, query, top_k=50):
        texts = [doc['text'] for doc in corpus]
        
        corpus_emb = torch.tensor([doc['embedding'] for doc in corpus])
        query_emb = torch.tensor(self.encode([query]))

        dot_scores = torch.matmul(query_emb, corpus_emb.T).squeeze()
        values, indices = torch.topk(dot_scores, k=top_k)
        # reranked_indices = self.rerank(query, texts, indices.tolist())
        
        # return reranked_indices
        return indices.tolist()
    
    def rerank(self, query, texts, retrieved_indices):
        num_indices = len(retrieved_indices)
        top_texts = [texts[i] for i in retrieved_indices]
        features = self.cross_tokenizer([query] * num_indices, top_texts, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            scores = self.cross_model(**features).logits
            scores = torch.flatten(scores)

        cross_values, cross_indices = torch.topk(scores, k=num_indices)
        return [retrieved_indices[i] for i in cross_indices.tolist()]
