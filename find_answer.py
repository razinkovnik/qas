import numpy as np
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForQuestionAnswering
from typing import List, Tuple
import pandas as pd


def context_score(tokenizer: BertTokenizer, model: BertForNextSentencePrediction, question: str, context: List[str], max_len=64):
    batch = tokenizer.batch_encode_plus(list(zip([question] * len(context), context)), max_length=max_len,
                                        truncation=True,
                                        pad_to_max_length=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**batch)[0]
    return np.array([a - b for a, b in output.cpu().numpy()])


def find_paragraph(tokenizer: BertTokenizer, model: BertForNextSentencePrediction, question: str, context: str,
                   max_len=256, batch_size=16):
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
        batch = tokenizer.batch_encode_plus(list(zip([question] * batch_size, parts[:batch_size])), max_length=max_len,
                                            truncation=True,
                                            pad_to_max_length=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(**batch)[0]
        results += [a - b for a, b in output.cpu().tolist()]
        parts = parts[batch_size:]
    return np.array(results), [tokenizer.decode(tokenizer.encode(part), skip_special_tokens=True) for part in all_parts]


def find_answer(tokenizer: BertTokenizer, answer_model: BertForQuestionAnswering, query: str, text: str) -> str:
    with torch.no_grad():
        start, end = answer_model(
            **tokenizer.encode_plus(query, text, max_length=256, truncation=True, return_tensors="pt"))
    start_pos = torch.argmax(start).item()
    end_pos = torch.argmax(end).item()
    if start_pos >= end_pos:
        start = torch.softmax(start, dim=1)
        end = torch.softmax(end, dim=1)
        k = -2
        start_args = torch.argsort(start).tolist()[0]
        end_args = torch.argsort(end).tolist()[0]
        calc_score = lambda start_pos, end_pos: start[0][start_pos] * end[0][end_pos]
        s_score, e_score = 0, 0
        s_pos, e_pos = start_pos, end_pos
        while s_score == 0 or e_score == 0:
            s_pos = start_args[k]
            e_pos = end_args[k]
            s_score = 0 if s_pos > end_pos else calc_score(s_pos, end_pos)
            e_score = 0 if e_pos < start_pos else calc_score(start_pos, e_pos)
            k -= 1
        if s_score > e_score:
            start_pos = s_pos
        else:
            end_pos = e_pos
    return tokenizer.decode(tokenizer.encode(query, text)[start_pos:end_pos])

