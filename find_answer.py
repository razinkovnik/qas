# %%

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction, BertForQuestionAnswering
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from string import punctuation
import re

# %%

vectors_path = "vectors.npy"
indexes_path = "indexes.npy"
model_name = "DeepPavlov/rubert-base-cased"
primary_task_path = "C:\\Users\\m31k0l2\\Google Диск\\quansw\\primary"
gold_task_path = "C:\\Users\\m31k0l2\\Google Диск\\quansw"


# %%

def setup(vectors_path: str, indexes_path: str, model_name: str, primary_task_path: str, gold_task_path: str) \
        -> Tuple[
            np.ndarray, List[int], BertTokenizer, BertModel, BertForNextSentencePrediction, BertForQuestionAnswering]:
    vectors = np.load(vectors_path)
    indexes = np.load(indexes_path)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    search_model = BertModel.from_pretrained(model_name)
    sent_model = BertForNextSentencePrediction.from_pretrained(primary_task_path)
    answer_model = BertForQuestionAnswering.from_pretrained(gold_task_path)
    search_model.eval()
    answer_model.eval()
    sent_model.eval()
    sent_model.cuda()
    return vectors, indexes, tokenizer, search_model, sent_model, answer_model


# %%

vectors, indexes, tokenizer, search_model, sent_model, answer_model = setup(vectors_path, indexes_path,
                                                                            model_name, primary_task_path,
                                                                            gold_task_path)

# %%

docs = pd.read_csv("dataset/docs.csv")


# %%

def search(tokenizer: BertTokenizer, search_model: BertModel, vectors: np.ndarray, indexes: List[int], query: str) -> \
List[int]:
    with torch.no_grad():
        out = search_model(input_ids=tokenizer.encode(query, return_tensors="pt", max_length=64, truncation=True))[0]
    query_vector = out.numpy()
    query_vector = np.average(query_vector, axis=1)
    dist = np.dot(vectors, query_vector.T).flatten()
    predicts = dist.argsort()[::-1]
    return list(dict.fromkeys([indexes[p] for p in predicts]))


# %%

def find_paragraph(model, question: str, context: str, max_len=256, batch_size=4):
    q_len = len(tokenizer.tokenize(question))
    context_tokens = tokenizer.tokenize(context)
    part_len = max_len - q_len - 3
    parts = []
    n = 0
    while n < len(context_tokens):
        parts += [context_tokens[n: n + part_len]]
        n += part_len // 2
    results = []
    all_parts = parts[:]
    while len(parts) > 0:
        batch = tokenizer.batch_encode_plus(zip([question] * len(parts), parts), max_len=max_len,
                                            pad_to_max_length=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**batch)[0]
        results += [a - b for a, b in output.cpu().tolist()]
        parts = parts[batch_size:]
    return results, all_parts


# %%

def search_paragraph(tokenizer: BertTokenizer, model: BertForNextSentencePrediction, query: str, context: str,
                     batch_size=16, block_size=256) -> Tuple[float, str]:
    q_len = len(tokenizer.tokenize(query))
    context_tokens = tokenizer.tokenize(context)
    part_len = block_size - q_len - 3
    parts = []
    n = 0
    while n < len(context_tokens):
        parts += [context_tokens[n: n + part_len]]
        n += part_len // 2
    all_parts = parts[:]

    results = []
    while len(parts) > 0:
        batch = parts[:batch_size]
        batch = tokenizer.batch_encode_plus(list(zip([query] * len(batch), batch)), max_length=block_size,
                                            truncation='only_second', pad_to_max_length=True, return_tensors="pt").to(
            "cuda")
        with torch.no_grad():
            output = model(**batch)[0]
        results += [a - b for a, b in output.tolist()]
        parts = parts[batch_size:]
    n = np.array(results).argmax()
    score = results[n]
    paragraph = tokenizer.decode(tokenizer.encode(all_parts[n]), skip_special_tokens=True)
    return score, paragraph


# %%

def search_answer(tokenizer, vectors, indexes, search_model, answer_model, query, max_count=10, min_score=4, start_from=0) -> Tuple[str, float]:
    predicts = search(tokenizer, search_model, vectors, indexes, query)[start_from:]
    text = ""
    count = 0
    best_score = -100.0
    for predict in predicts:
        count += 1
        score, paragraph = search_paragraph(tokenizer, sent_model, query, docs.docs[predict], block_size=64)
        if score > best_score:
            best_score = score
            text = paragraph
        if score > min_score:
            text = paragraph
            break
        if max_count - start_from == count:
            break
    with torch.no_grad():
        start, end = answer_model(**tokenizer.encode_plus(query, text, max_len=256, return_tensors="pt"))
    start = torch.argmax(start).item()
    end = torch.argmax(end).item()
    return tokenizer.decode(tokenizer.encode(query, text)[start:end]), best_score


# %%

n = 53
query = docs.questions[n]
print(query)

# %%

for i in range(0, 50, 10):
    answer, score = search_answer(tokenizer, vectors, indexes, search_model, answer_model, query, min_score=4,
                                  max_count=i + 10, start_from=i)
    if score < 0:
        break
    print(f"{i // 10 + 1})", answer, score)
    if len(answer) > 1 and score > i // 10:
        break

