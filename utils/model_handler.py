import os
import json
import uuid
from tqdm import tqdm, trange
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)
import numpy as np
import faiss

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import en_core_web_md


class ModelHandler:
    encoder_save_path = "models/splade_v2_max"
    cross_save_path = "models/ms-marco-MiniLM-L-12-v2"
    qa_save_path = "models/tinyroberta-squad2"

    INDEX_PATH = "index/"
    INFO_PATH = "info/"

    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            ModelHandler.encoder_save_path
        )
        self.encoder_model = AutoModelForMaskedLM.from_pretrained(
            ModelHandler.encoder_save_path
        )
        self.encoder_model.eval()

        # self.cross_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.cross_save_path)
        # self.cross_model = AutoModelForSequenceClassification.from_pretrained(ModelHandler.cross_save_path)
        # self.cross_model.eval()

        # self.qa_tokenizer = AutoTokenizer.from_pretrained(ModelHandler.qa_save_path)
        # self.qa_model = AutoModelForQuestionAnswering.from_pretrained(ModelHandler.qa_save_path)
        # self.qa = pipeline("question-answering", model=self.qa_model, tokenizer=self.qa_tokenizer)

        self.id2token = {
            id: token for token, id in self.encoder_tokenizer.get_vocab().items()
        }
        self.nlp = en_core_web_md.load()

    def numpy_encode(self, texts, chunk_size=25):
        if isinstance(texts, str):
            texts = [texts]

        print(f"encoding {len(texts)} texts")
        embeddings = torch.empty(0)
        with torch.no_grad():
            for i in trange(0, len(texts), chunk_size):
                chunk = texts[i : i + chunk_size]
                tokens = self.encoder_tokenizer(
                    chunk, return_tensors="pt", padding=True, truncation=True
                )
                output = self.encoder_model(**tokens)
                vecs = torch.sum(
                    torch.log(1 + torch.relu(output["logits"]))
                    * tokens["attention_mask"].unsqueeze(-1),
                    dim=1,
                )
                embeddings = torch.cat((embeddings, vecs))
        print("finished encoding texts")
        return embeddings.cpu().numpy()  # .tolist()

    def create_index(self, file_id, texts):
        embeddings = self.numpy_encode(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        serialized_index = faiss.serialize_index(index)
        np.save("index/" + file_id + ".npy", serialized_index)

    def retrieve(self, file_id, query, top_k=50):
        np_obj = np.load("index/" + file_id + ".npy")
        index = faiss.deserialize_index(np_obj)
        top_k = min(index.ntotal, top_k)

        q_emb = self.numpy_encode(query)
        results = index.search(q_emb, top_k)
        indices = results[1].flatten()

        # reranked_indices = self.rerank(query, texts, indices.tolist())
        # return reranked_indices
        return indices.tolist()

    def keywords(self, query, context, top_k=5, thresh=0.5):
        q_words = self.preprocess_text(query)
        c_words = self.preprocess_text(context)

        keywords = []
        for q_word in q_words:
            for c_word in c_words:
                q_token = self.nlp(q_word)
                c_token = self.nlp(c_word)
                sim = q_token.similarity(c_token)
                if sim >= thresh:
                    keywords.append(c_word)
        return list(set(keywords[:top_k]))

    def preprocess_text(self, text, min_chars=3):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in string.punctuation]
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [token for token in tokens if len(token) >= min_chars]
        return tokens
