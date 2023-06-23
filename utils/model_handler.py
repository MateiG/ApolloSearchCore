import os
import json
import uuid
from tqdm import tqdm, trange
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

class ModelHandler():
    encoder_save_path = 'models/all-mpnet-base-v2' # 'models/all-MiniLM-L6-v2'
    cross_save_path = 'models/ms-marco-MiniLM-L-12-v2' # 'models/ms-marco-MiniLM-L-2-v2'
    qa_save_path = 'models/tinyroberta-squad2'

    INDEX_PATH='index/'

    def __init__(self):
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.encoder_save_path, )
        self.encoder_model = AutoModel.from_pretrained(ModelHandler.encoder_save_path)
        self.encoder_model.eval()

        self.cross_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.cross_save_path)
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(ModelHandler.cross_save_path)
        self.cross_model.eval()

        self.qa_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.qa_save_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(ModelHandler.qa_save_path)
        self.qa = pipeline('question-answering', model=self.qa_model, tokenizer=self.qa_tokenizer)

        self.cos_sim = CosineSimilarity()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, documents, chunk_size=100):
        embeddings = []
        with torch.no_grad():
            for chunk_start in trange(0, len(documents), chunk_size):
                chunk = documents[chunk_start:chunk_start+chunk_size]
                encoded_input = self.encoder_tokenizer(chunk, padding=True, truncation=True, return_tensors='pt')
                model_output = self.encoder_model(**encoded_input)

                chunk_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

                chunk_embeddings = chunk_embeddings.cpu().numpy().tolist()
                embeddings.extend(chunk_embeddings)
        
        return embeddings
    
    def highlight(self, query, context, chars: int = 50):
        res = self.qa({'question': query, 'context': context})
        start = max(0, res['start'] - chars)
        end = min(res['end'] + chars, len(context))

        return start, end # res['answer']
    
    def retrieve(self, corpus, query, top_k=50):
        texts = [doc['text'] for doc in corpus]
        
        corpus_emb = torch.tensor([doc['embedding'] for doc in corpus])
        query_emb = torch.tensor(self.encode([query]))

        similarities = self.cos_sim(query_emb, corpus_emb)
        values, indices = torch.topk(similarities, k=top_k)
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
