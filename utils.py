from collections import namedtuple
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from typing import List, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
import pandas as pd
from tqdm import tqdm as tqdm_base


class QAData:
    def __init__(self, tokenizer: PreTrainedTokenizer, context: str, question: str, answer_start: int, answer_text: str):
        self.context = context
        self.question = question
        start_str = context[:answer_start]
        self.start = len(tokenizer.tokenize(start_str)) + 1
        self.end = self.start + len(tokenizer.tokenize(answer_text))


Batch = namedtuple(
    "Batch", ["input_ids", "token_type_ids", "attention_mask", "start", "end"]
)


class MyDataset(Dataset):
    def __init__(self, data: List[QAData]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def df2qas(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> List[QAData]:
    qas_lst: List[QAData] = []
    for i in range(len(df)):
        _, context, question, answer_start, answer_text = df.iloc[i]
        qas_lst += [QAData(tokenizer, context, question, answer_start, answer_text)]
    return qas_lst


def collate(data: List[QAData], tokenizer: PreTrainedTokenizer, block_size: int) -> Batch:
    context_lst = [qas.context for qas in data]
    question_lst = [qas.question for qas in data]
    start_lst = [qas.start for qas in data]
    end_lst = [qas.end for qas in data]
    encode_data = [tokenizer.encode_plus(ctx, q, max_length=block_size, return_tensors="pt", truncation=True, padding=True) for ctx, q in zip(context_lst, question_lst)]
    input_ids = torch.stack([item["input_ids"].squeeze(0) for item in encode_data])
    token_type_ids = torch.stack([item["token_type_ids"].squeeze(0) for item in encode_data])
    attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in encode_data])
    return Batch(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start=torch.tensor(start_lst), end=torch.tensor(end_lst))


def build_data_iterator(tokenizer: PreTrainedTokenizer, dataset: List[QAData], batch_size: int, block_size: int, random_sampler=False) -> DataLoader:
    sampler = RandomSampler(dataset) if random_sampler else SequentialSampler(dataset)
    iterator = DataLoader(
        MyDataset(dataset), sampler=sampler, batch_size=batch_size, collate_fn=lambda data: collate(data, tokenizer, block_size),
    )
    return iterator


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)
