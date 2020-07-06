from primary_task import read_data
import pandas as pd
from random import sample


def grab_questions_yes_no(filename: str) -> pd.DataFrame:
    data = read_data(f"dataset/{filename}.jsonl")
    questions = []
    answers = []
    for item in data:
        questions += [item['question_text']]
        answers += [item['annotations'][0]['yes_no_answer']]
    data = pd.DataFrame({
        "questions": questions,
        "answers": answers
    })
    data = pd.DataFrame(data)
    data.to_csv(f"dataset/{filename}.csv")
    return data


def grab_primary_task_yes_no(filename: str) -> pd.DataFrame:
    data = read_data(f"dataset/{filename}.jsonl")
    parts, titles, questions, answers = [], [], [], []
    for item in data:
        annotations = dict([(a['passage_answer']['candidate_index'], a['yes_no_answer']) for a in item['annotations']])
        if set(annotations.values()) == {'NONE'}:
            continue
        answer_candidates = item['passage_answer_candidates']
        question = item['question_text']
        text = item['document_plaintext']
        title = item['document_title']
        for i in sample([i for i in range(len(answer_candidates)) if annotations.keys()], len(annotations)):
            annotations[i] = 'NONE'
        for i, answer in annotations.items():
            part = answer_candidates[i]
            start = part['plaintext_start_byte']
            end = part['plaintext_end_byte']
            parts += [text.encode('utf-8')[start:end].decode("utf-8")]
            titles += [title]
            questions += [question]
            answers += [answer]
    data = pd.DataFrame({
        "parts": parts,
        "titles": titles,
        "questions": questions,
        "answers": answers
    })
    data.to_csv(f"dataset/{filename}.csv")
    return data


if __name__ == "__main__":
    grab_primary_task_yes_no("train")
    print("train - ok")
    grab_primary_task_yes_no("test")
    print("test - ok")
