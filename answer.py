import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import argparse


def find_answer(context: str, question: str):
    input_data = tokenizer.encode_plus(question, context, return_tensors="pt")
    with torch.no_grad():
        out = model(**input_data)
    start, end = out[0], out[1]
    start = torch.argmax(start).item()
    end = torch.argmax(end).item()
    return tokenizer.decode(tokenizer.encode(context)[start:end])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели')
    parser.add_argument('--model', type=str, default="DeepPavlov/rubert-base-cased", help='модель')
    parser.add_argument('--model_dir', type=str, default="models", help='путь к модели')
    parser.add_argument('--context', type=str)
    parser.add_argument('--question', type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    answer = find_answer(args.context, args.question)
    print(context)
    print("="*10)
    print(question)
    print(answer)
