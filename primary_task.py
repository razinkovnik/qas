import json
from random import shuffle
from transformers import AutoTokenizer, AdamW, BertForNextSentencePrediction
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from typing import List
from collections import namedtuple
from args import TrainingArguments
from torch.utils.tensorboard import SummaryWriter
import logging


Batch = namedtuple(
    "Batch", ["input_data", "labels"]
)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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


def collate(data: List, tokenizer: AutoTokenizer, block_size: int) -> Batch:
    labels = [item['label'] for item in data]
    questions = [item['question'] for item in data]
    texts = [item['text'] for item in data]
    input_data = tokenizer.batch_encode_plus(list(zip(questions, texts)), max_length=block_size, truncation=True, padding=True, return_tensors="pt")
    return Batch(input_data, torch.tensor(labels))


def build_data_iterator(tokenizer: AutoTokenizer, items: List, batch_size: int, block_size: int) -> DataLoader:
    dataset = MyDataset(items)
    iterator = DataLoader(
        dataset, sampler=RandomSampler(dataset), batch_size=batch_size, collate_fn=lambda data: collate(data, tokenizer, block_size),
    )
    return iterator


def evaluate(model: BertForNextSentencePrediction, iterator: DataLoader, args: TrainingArguments) -> float:
    model.eval()
    model.to(args.device)
    total = 0
    for batch, labels in tqdm(iterator, desc='eval'):
        with torch.no_grad():
            loss = model(**input_data.to(args.device), next_sentence_label=labels.to(args.device))[0]
        total += loss.item()
    model.train()
    return total / len(iterator)


def train_epoch(model: BertForNextSentencePrediction, optimizer: torch.optim.Optimizer, tr_iterator: DataLoader, ev_iterator: DataLoader, args: TrainingArguments, writer: SummaryWriter, logger: logging.Logger, n: int):
    model.to(args.device)
    model.train()
    step = 0
    train_loss = 0
    if args.evaluate_during_training:
        loss = evaluate(model, ev_iterator, args)
        logger.info(f"eval loss: {loss}")
        writer.add_scalar('Loss/eval', loss, step)
    for batch in tr_iterator:
        loss = model(**batch.input_data.to(args.device), next_sentence_label=batch.labels.to(args.device))[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        model.zero_grad()
        writer.add_scalar('Loss/train', loss.item(), step)
        step += 1
        if step % args.save_steps == 0:
            model.save_pretrained(args.output_dir)
            logger.info(f"epoch: {n + step / len(tr_iterator)}")
            logger.info(f"train loss: {train_loss / args.save_steps}")
            train_loss = 0
            if args.evaluate_during_training:
                loss = evaluate(model, ev_iterator, args)
                logger.info(f"eval loss: {loss}")
                writer.add_scalar('Loss/eval', loss, step)
        logger.info(f"train loss: {train_loss / args.save_steps}")
        if args.evaluate_during_training:
            loss = evaluate(model, ev_iterator, args)
            logger.info(f"eval loss: {loss}")
            writer.add_scalar('Loss/eval', loss, step)
        model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    data = read_data("dataset/tydiqa.jsonl")
    items = [part for parts in [json2data(item) for item in data] for part in parts]
    negative_items = [item for item in items if item['label'] == 1]
    positive_items = [item for item in items if item['label'] == 0]
    items = positive_items + negative_items[:len(positive_items)]
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = BertForNextSentencePrediction.from_pretrained("DeepPavlov/rubert-base-cased")
    args = TrainingArguments()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    iterator = build_data_iterator(tokenizer, items, args.train_batch_size, args.block_size)
    # tr_iterator = build_data_iterator(tokenizer, tr_ds, args.train_batch_size, args.block_size)
    # ev_iterator = build_data_iterator(tokenizer, ev_ds, args.eval_batch_size, args.block_size)
    num_train_epochs = 1
    logger = logging.getLogger("prim_qas")
    writer = SummaryWriter(log_dir=args.log_dir)
    for i in range(num_train_epochs):
        train_epoch(model, optimizer, tr_iterator, ev_iterator, args, writer, logger, i)
