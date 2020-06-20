import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, BertForQuestionAnswering
import pandas as pd
from utils import df2qas, build_data_iterator, Batch, tqdm
from torch.utils.data import DataLoader
import logging
from torch.utils.tensorboard import SummaryWriter
from args import TrainingArguments
import argparse


def evaluate(model: BertForQuestionAnswering, iterator: DataLoader, args: TrainingArguments) -> float:
    model.eval()
    model.to(args.device)
    total = 0
    for input_ids, token_type_ids, attention_mask, start, end in tqdm(iterator, desc='eval'):
        with torch.no_grad():
            loss = model(input_ids.to(args.device),
                         token_type_ids=token_type_ids.to(args.device),
                         attention_mask=attention_mask.to(args.device),
                         start_positions=start.to(args.device),
                         end_positions=end.to(args.device)
                         )[0]
        total += loss.item()
    model.train()
    return total / len(iterator)


def train_epoch(model: BertForQuestionAnswering, optimizer: torch.optim.Optimizer, tr_iterator: DataLoader, ev_iterator: DataLoader,
                args: TrainingArguments, writer: SummaryWriter, logger: logging.Logger, n: int):
    model.to(args.device)
    model.train()
    step = 0
    train_loss = 0
    if args.evaluate_during_training:
        loss = evaluate(model, ev_iterator, args)
        logger.info(f"eval loss: {loss}")
        writer.add_scalar('Loss/eval', loss, step)
    for input_ids, token_type_ids, attention_mask, start, end in tqdm(tr_iterator, desc='train'):
        loss = model(input_ids.to(args.device),
                     token_type_ids=token_type_ids.to(args.device),
                     attention_mask=attention_mask.to(args.device),
                     start_positions=start.to(args.device),
                     end_positions=end.to(args.device)
                     )[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        model.zero_grad()
        writer.add_scalar('Loss/train', loss.item(), step)
        train_loss += loss.item()
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
    parser = argparse.ArgumentParser(description='Обучение модели')
    parser.add_argument('--dataset_path', type=str, default="dataset/gold.csv", help='путь к датасету')
    parser.add_argument('--model', type=str, default="xlm-mlm-xnli15-1024", help='модель')
    parser.add_argument('--load', help='загрузить модель', action='store_true')
    parser.add_argument('--output_dir', type=str, default="models", help='путь к модели')
    parser.add_argument('--train_data_size', type=float, default=0.95, help='относительный размер данных для тренировки')
    parser.add_argument('--train_batch_size', type=int, default=4, help='размер тренировочного батча')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='размер тестового батча')
    parser.add_argument('--block_size', type=int, default=256, help='размер блока текста')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='количество эпох')
    parser.add_argument('--save_steps', type=int, default=100, help='шаг сохранения')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser_args = parser.parse_args()
    args = TrainingArguments()
    args.output_dir = parser_args.output_dir
    args.train_batch_size = parser_args.train_batch_size
    args.eval_batch_size = parser_args.eval_batch_size
    args.block_size = parser_args.block_size
    args.save_steps = parser_args.save_steps
    args.learning_rate = parser_args.lr
    args.device = parser_args.device

    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(parser_args.model)
    if parser_args.load:
        model = AutoModelForQuestionAnswering.from_pretrained(parser_args.output_dir)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(parser_args.model)
    df = pd.read_csv(parser_args.dataset_path)
    data = df2qas(df, tokenizer)
    data = [item for item in data if item.end < args.block_size - len(tokenizer.tokenize(item.question))]
    n = int(len(data) * parser_args.train_data_size)
    tr_ds, ev_ds = data[:n], data[n:]
    tr_iterator = build_data_iterator(tokenizer, tr_ds, args.train_batch_size, args.block_size)
    ev_iterator = build_data_iterator(tokenizer, ev_ds, args.eval_batch_size, args.block_size)
    logger = logging.getLogger("qas")
    writer = SummaryWriter(log_dir=args.log_dir)
    if parser_args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    for i in range(parser_args.num_train_epochs):
        train_epoch(model, optimizer, tr_iterator, ev_iterator, args, writer, logger, i)
