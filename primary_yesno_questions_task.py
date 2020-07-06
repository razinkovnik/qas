import json
from random import shuffle
from utils import *
from transformers import BertTokenizer, AdamW, BertForSequenceClassification
import torch
from typing import List
from args import TrainingArguments
import pandas as pd


def collate(data: List, tokenizer: BertTokenizer, block_size: int) -> Dict:
    questions = [item[0] for item in data]
    labels = [0 if item[1] == 'NONE' else 1 for item in data]
    input_data = tokenizer.batch_encode_plus(questions, max_length=block_size,
                                             truncation=True, pad_to_max_length=True, return_tensors="pt").to(
        args.device)
    input_data['labels'] = torch.tensor(labels).to(args.device)
    return input_data


def load_data(filename: str, tokenizer: BertTokenizer, batch_size: int, args: TrainingArguments) -> DataLoader:
    df = pd.read_csv(filename)
    yes_no_answers = df[df['answers'] != 'NONE']
    none_answers = df[df['answers'] == 'NONE'].sample(n=len(yes_no_answers), random_state=2)
    df = yes_no_answers.append(none_answers, ignore_index=True)
    items = list(zip(df.questions.to_list(), df.answers.to_list()))
    return build_data_iterator(items, batch_size, lambda data: collate(data, tokenizer, args.block_size))


if __name__ == "__main__":
    args = setup()
    args.train_dataset = "dataset/train.csv"
    args.test_dataset = "dataset/test.csv"
    # args.load = True
    args.num_train_epochs = 1
    tokenizer, model, optimizer = init_model(BertForSequenceClassification, args)
    # train(tokenizer, model, optimizer, load_data, args)
