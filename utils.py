from collections import namedtuple
from tqdm import tqdm as tqdm_base
import logging
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertPreTrainedModel
from typing import List, Callable
from args import TrainingArguments
import torch
import argparse
import math


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


logger = logging.getLogger("qas")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

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


def build_data_iterator(items: List, batch_size: int, collate: Callable[[List], Batch]) -> DataLoader:
    dataset = MyDataset(items)
    iterator = DataLoader(
        dataset, sampler=RandomSampler(dataset), batch_size=batch_size,
        collate_fn=collate,
    )
    return iterator


def evaluate(model: BertPreTrainedModel, iterator: DataLoader, args: TrainingArguments,
             evaluate_batch: Callable[[Batch], float]) -> float:
    model.eval()
    model.to(args.device)
    total = []
    for batch in tqdm(list(iterator), desc='eval'):
        total += [evaluate_batch(batch)]
    model.train()
    total = [x for x in total if not math.isnan(x)]
    return sum(total) / len(total)


def train_epoch(model: BertPreTrainedModel, optimizer: torch.optim.Optimizer, iterator: DataLoader,
                args: TrainingArguments, train_batch: Callable[[Batch], torch.Tensor],
                evaluate: Callable[[], float], num_epoch=0):
    model.train()
    train_loss = []
    logger.info(f"eval loss: {evaluate()}")
    for step, batch in enumerate(tqdm(iterator, desc="train")):
        loss = train_batch(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        model.zero_grad()
        train_loss += [loss.item()]
        if args.writer:
            args.writer.add_scalar('Loss/train', loss.item(), num_epoch * len(iterator) + step)
        if step > 0 and step % args.save_steps == 0:
            model.save_pretrained(args.output_dir)
            logger.info(f"epoch: {num_epoch + step / len(iterator)}")
            tr_loss = [x for x in train_loss if not math.isnan(x)]
            logger.info(f"train loss: {sum(tr_loss[-args.save_steps:]) / args.save_steps}")
            if args.writer:
                loss = evaluate()
                logger.info(f"eval loss: {loss}")
                args.writer.add_scalar('Loss/eval', loss, step)
    logger.info(f"eval loss: {evaluate()}")
    model.save_pretrained(args.output_dir)


def setup() -> TrainingArguments:
    args = TrainingArguments()
    parser = argparse.ArgumentParser(description='Обучение модели')
    parser.add_argument('--train_dataset', default="dataset/tydiqa.json", type=str, help='путь к тренировочному датасету')
    parser.add_argument('--test_dataset', default="dataset/tydiqa.json", type=str, help='путь к тестовому датасету')
    parser.add_argument('--model', type=str, default="DeepPavlov/rubert-base-cased", help='модель')
    parser.add_argument('--load', help='загрузить модель', action='store_true')
    parser.add_argument('--output_dir', type=str, default="models", help='путь к модели')
    parser.add_argument('--train_batch_size', type=int, default=4, help='размер тренировочного батча')
    parser.add_argument('--test_batch_size', type=int, default=4, help='размер тестового батча')
    parser.add_argument('--block_size', type=int, default=128, help='размер блока текста')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='количество эпох')
    parser.add_argument('--save_steps', type=int, default=100, help='шаг сохранения')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default="cpu")
    parser_args = parser.parse_args()
    args.model_name = parser_args.model
    args.train_dataset = parser_args.train_dataset
    args.test_dataset = parser_args.test_dataset
    args.load = parser_args.load
    args.learning_rate = parser_args.lr
    args.save_steps = parser_args.save_steps
    args.device = parser_args.device
    args.block_size = parser_args.block_size
    args.train_batch_size = parser_args.train_batch_size
    args.test_batch_size = parser_args.test_batch_size
    args.output_dir = parser_args.output_dir
    args.num_train_epochs = parser_args.num_train_epochs
    return args
