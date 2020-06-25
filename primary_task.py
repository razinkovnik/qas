import json
from random import shuffle
from utils import *
from transformers import BertTokenizer, AdamW, BertForNextSentencePrediction
import torch
from typing import List
from args import TrainingArguments


def read_data(filename: str):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    data = [json.loads(js) for js in lines]
    return [el for el in data if el['language'] == 'russian']


def json2data(item):
    context = item['document_plaintext'].encode("utf-8")
    nums = set([annotation['passage_answer']['candidate_index'] for annotation in item['annotations']])
    question = item['question_text']
    parts = []
    for i, part_text in enumerate(item['passage_answer_candidates']):
        start, end = part_text['plaintext_start_byte'], part_text['plaintext_end_byte']
        text = context[start:end].decode("utf-8")
        parts += [{'question': question, 'text': text, 'label': 0 if i in nums else 1}]
    return parts


def collate(data: List, tokenizer: BertTokenizer, block_size: int) -> Dict:
    labels = [item['label'] for item in data]
    questions = [item['question'] for item in data]
    texts = [item['text'] for item in data]
    input_data = tokenizer.batch_encode_plus(list(zip(questions, texts)), max_length=block_size,
                                             truncation='only_second', pad_to_max_length=True, return_tensors="pt").to(
        args.device)
    input_data['next_sentence_label'] = torch.tensor(labels).to(args.device)
    return input_data


def load_data(filename: str, tokenizer: BertTokenizer, batch_size: int, args: TrainingArguments) -> DataLoader:
    data = read_data(filename)
    items = [part for parts in [json2data(item) for item in data] for part in parts]
    negative_items = [item for item in items if item['label'] == 1]
    positive_items = [item for item in items if item['label'] == 0]
    shuffle(negative_items)
    items = positive_items + negative_items[:len(positive_items)]
    return build_data_iterator(items, batch_size, lambda data: collate(data, tokenizer, args.block_size))


if __name__ == "__main__":
    args = setup()
    # args.train_dataset = "dataset/tydiqa.jsonl"
    # args.test_dataset = "dataset/tydiqa.jsonl"
    tokenizer, model, optimizer = init_model(BertForNextSentencePrediction, args)
    train(tokenizer, model, optimizer, load_data, args)
