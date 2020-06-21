import json
from transformers import AutoTokenizer, AdamW, BertForQuestionAnswering
import torch
from typing import List, Callable
from args import TrainingArguments
from utils import *
from torch.utils.data import DataLoader


def read_data(filename: str, tokenizer: AutoTokenizer) -> List:
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
        ds += [{
            "question": question,
            "context": context,
            "start": start,
            "end": end
        }]
    return ds


def collate(data: List, tokenizer: AutoTokenizer, block_size: int) -> Batch:
    starts = [item['start'] for item in data]
    ends = [item['end'] for item in data]
    questions = [item['question'] for item in data]
    contexts = [item['context'] for item in data]
    input_data = tokenizer.batch_encode_plus(list(zip(questions, contexts)), max_length=block_size, truncation=True,
                                             padding=True, return_tensors="pt")
    return Batch(input_data, (torch.tensor(starts), torch.tensor(ends)))


def evaluate_batch(model: BertForQuestionAnswering, batch: Batch) -> float:
    start, end = batch.labels
    try:
        with torch.no_grad():
            loss = model(**batch.input_data.to(args.device), start_positions=start.to(args.device),
                         end_positions=end.to(args.device))[0]
        return loss.item()
    except IndexError:
        return math.nan


def train_batch(model: BertForQuestionAnswering, batch: Batch) -> torch.Tensor:
    start, end = batch.labels
    return model(**batch.input_data.to(args.device), start_positions=start.to(args.device),
                 end_positions=end.to(args.device))[0]


def load_data(filename: str, tokenizer: AutoTokenizer, batch_size: int, args: TrainingArguments) -> DataLoader:
    data = read_data(filename, tokenizer)
    data = [item for item in data if item['end'] < args.block_size - 1]
    return build_data_iterator(data, batch_size, lambda data: collate(data, tokenizer, args.block_size))


if __name__ == "__main__":
    args = setup()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tr_iterator = load_data(args.train_dataset, tokenizer, args.train_batch_size, args)
    ev_iterator = load_data(args.test_dataset, tokenizer, args.test_batch_size, args)
    if args.load:
        model = BertForQuestionAnswering.from_pretrained(args.output_dir)
    else:
        model = BertForQuestionAnswering.from_pretrained(args.model_name)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    for n in range(args.num_train_epochs):
        train_epoch(model, optimizer, tr_iterator, args,
                    lambda batch: train_batch(model, batch),
                    lambda: evaluate(model, ev_iterator, args, lambda batch: evaluate_batch(model, batch)), n)
