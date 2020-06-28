import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import argparse


def find_answer(tokenizer: BertTokenizer, model: BertForQuestionAnswering, context: str, question: str):
    input_data = tokenizer.encode_plus(question, context, return_tensors="pt")
    with torch.no_grad():
        out = model(**input_data)
    start, end = out[0], out[1]
    start = torch.argmax(start).item()
    end = torch.argmax(end).item()
    return tokenizer.decode(tokenizer.encode(question, context)[start:end])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели')
    parser.add_argument('--model', type=str, default="DeepPavlov/rubert-base-cased", help='модель')
    parser.add_argument('--model_dir', type=str, default="models", help='путь к модели')
    parser.add_argument('--context', type=str)
    parser.add_argument('--question', type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertForQuestionAnswering.from_pretrained(args.model_dir)
    model.eval()
    print(find_answer(tokenizer, model, args.context, args.question))