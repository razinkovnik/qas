import json
from random import shuffle
from utils import *
from transformers import BertTokenizer, AdamW, BertForSequenceClassification
import torch
from typing import List
from args import TrainingArguments
import pandas as pd


def collate(data: List, tokenizer: BertTokenizer, block_size: int) -> Dict:
    questions = [item[3] for item in data]
    texts = [item[1] for item in data]
    label2id = {
        'YES': 1,
        'NO': 0,
        'NONE': 2
    }
    labels = [label2id[item[-1]] for item in data]
    input_data = tokenizer.batch_encode_plus(list(zip(questions, texts)), max_length=block_size,
                                             truncation='only_second', pad_to_max_length=True, return_tensors="pt").to(
        args.device)
    input_data['labels'] = torch.tensor(labels).to(args.device)
    return input_data


def load_data(filename: str, tokenizer: BertTokenizer, batch_size: int, args: TrainingArguments) -> DataLoader:
    df = pd.read_csv(filename)
    return build_data_iterator(df.values.tolist(), batch_size, lambda data: collate(data, tokenizer, args.block_size))


if __name__ == "__main__":
    args = setup()
    args.train_dataset = "dataset/train.csv"
    args.test_dataset = "dataset/test.csv"
    args.block_size = 256
    # args.load = True
    args.num_train_epochs = 1
    tokenizer, model, optimizer = init_model(BertForSequenceClassification, args, labels=3)
    train(tokenizer, model, optimizer, load_data, args)
