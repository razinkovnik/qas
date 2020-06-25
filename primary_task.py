import json
from random import shuffle
from utils import *
from transformers import BertTokenizer, AdamW, BertForNextSentencePrediction
import torch
from typing import List
from args import TrainingArguments
import pandas as pd


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
    questions = [item[0] for item in data]
    texts = [item[1] for item in data]
    labels = [item[2] for item in data]
    input_data = tokenizer.batch_encode_plus(list(zip(questions, texts)), max_length=block_size,
                                             truncation='only_second', pad_to_max_length=True, return_tensors="pt").to(
        args.device)
    input_data['next_sentence_label'] = torch.tensor(labels).to(args.device)
    return input_data


def json2df(input_file: str, output_file: str):
    data = read_data(input_file)
    items = [part for parts in [json2data(item) for item in data] for part in parts]
    questions = [item['question'] for item in items]
    texts = [item['text'] for item in items]
    labels = [item['label'] for item in items]
    df = pd.DataFrame({"questions": questions, "texts": texts, "labels": labels})
    df.to_csv(output_file)


def load_data(filename: str, tokenizer: BertTokenizer, batch_size: int, args: TrainingArguments) -> DataLoader:
    df = pd.read_csv(filename)
    items = list(zip(df.questions.to_list(), df.texts.to_list(), df.labels.to_list()))
    negative_items = [item for item in items if item[2] == 1]
    positive_items = [item for item in items if item[2] == 0]
    shuffle(negative_items)
    items = positive_items + negative_items[:len(positive_items)]
    return build_data_iterator(items, batch_size, lambda data: collate(data, tokenizer, args.block_size))


if __name__ == "__main__":
    args = setup()
    # args.train_dataset = "dataset/primary_train.csv"
    # args.test_dataset = "dataset/primary_test.csv"
    tokenizer, model, optimizer = init_model(BertForNextSentencePrediction, args)
    train(tokenizer, model, optimizer, load_data, args)
