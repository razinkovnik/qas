import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
from utils import df2qas


def find_answer(context: str, question: str):
    code = tokenizer.encode_plus(context, question, return_tensors="pt")
    with torch.no_grad():
        out = model(code["input_ids"], token_type_ids=code["token_type_ids"], attention_mask=code["attention_mask"])
    start, end = out[0], out[1]
    start = torch.argmax(start).item()
    end = torch.argmax(end).item()
    return tokenizer.decode(tokenizer.encode(context)[start:end])


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModelForQuestionAnswering.from_pretrained("C:\\Users\\m31k0l2\\Google Диск\\quansw")
    df = pd.read_csv("dataset/gold.csv")
    data = df2qas(df, tokenizer)
    context = data[-1].context
    question = data[-1].question
    answer = find_answer(context, question)
    print(context)
    print("="*10)
    print(question)
    print(answer)
