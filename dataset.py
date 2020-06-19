import json
import pandas as pd
import argparse


def normalize_str(text):
    return "".join([chr(n) for n in [ord(x) for x in text] if not n == 769])


def convert_data(src, output):
    with open(src, encoding="utf-8") as f:
        data = f.readlines()
    gold_dataset = json.loads(data[0])
    rus_data = []
    counter = 0
    for i, item in enumerate(gold_dataset['data']):
        item = item['paragraphs'][0]
        qas = item['qas'][0]
        lang = qas['id']
        if 'russian' in lang:
            context = item['context']
            question = qas['question']
            answer = qas['answers'][0]
            answer_text = answer['text']
            answer_start = answer['answer_start']
            # answer_end = answer_start + len(answer_text)
            rus_data += [{
                'context': normalize_str(context),
                'question': normalize_str(question),
                'answer_start': answer_start,
                # 'answer_end': answer_end,
                'answer_text': answer_text
            }]
            counter += 1
    df = pd.DataFrame(rus_data)
    df.to_csv(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Подготовка датасета')
    parser.add_argument('--input_path', type=str, default="dataset/tydiqa-goldp-v1.1-dev.json", help='путь к json')
    parser.add_argument('--output_path', type=str, default="dataset/gold.csv", help='путь к датасету')
    args = parser.parse_args()
    convert_data(args.input_path, args.output_path)
