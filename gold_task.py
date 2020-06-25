import json
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from typing import List, Dict
from args import TrainingArguments
from utils import *
from torch.utils.data import DataLoader


def read_data(filename: str, tokenizer: BertTokenizer, args: TrainingArguments) -> List:
    with open(filename, encoding="utf-8") as f:
        data = f.readlines()
    items = [item for item in json.loads(data[0])['data'] if 'russian' in item['paragraphs'][0]['qas'][0]['id']]
    ds = []
    for item in items:
        paragraph = item['paragraphs'][0]
        context = paragraph['context']
        qas = paragraph['qas'][0]
        question = qas['question']
        answer = qas['answers'][0]
        answer_start = answer['answer_start']
        answer_text = answer['text']
        ids = tokenizer.encode(question, context[:answer_start])
        start = len(ids) - 1
        end = start + len(tokenizer.tokenize(answer_text))
        if end < args.block_size:
            ds += [{
                "question": question,
                "context": context,
                "start": start,
                "end": end
            }]
    return ds


def collate(data: List, tokenizer: BertTokenizer, block_size: int) -> Dict:
    starts = [item['start'] for item in data]
    ends = [item['end'] for item in data]
    questions = [item['question'] for item in data]
    contexts = [item['context'] for item in data]
    input_data = tokenizer.batch_encode_plus(list(zip(questions, contexts)), max_length=block_size,
                                             truncation='only_second', pad_to_max_length=True, return_tensors="pt").to(
        args.device)
    input_data['start_positions'] = torch.tensor(starts).to(args.device)
    input_data['end_positions'] = torch.tensor(ends).to(args.device)
    return input_data


def load_data(filename: str, tokenizer: BertTokenizer, batch_size: int, args: TrainingArguments) -> DataLoader:
    data = read_data(filename, tokenizer, args)
    return build_data_iterator(data, batch_size, lambda data: collate(data, tokenizer, args.block_size))


if __name__ == "__main__":
    args = setup()
    tokenizer, model, optimizer = init_model(BertForQuestionAnswering, args)
    train(tokenizer, model, optimizer, load_data, args)

