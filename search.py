from tfidf import load, search
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForQuestionAnswering, BertForSequenceClassification
import pandas as pd
from find_answer import context_score, find_paragraph, find_answer, find_yesno_answer, get_question_type
from typing import List


def search_context(tfidf, tokenizer: BertTokenizer, model: BertForNextSentencePrediction, texts: List[str],
                   question: str, max_length=1, batch_size=16):
    indexes, scores = search(tfidf, question)
    if scores[indexes[0]] / scores[indexes[1]] > 2:
        return indexes[0]
    for i in range(0, max_length * batch_size, batch_size):
        context = [texts[j] for j in indexes[i: i + batch_size]]
        score = context_score(tokenizer, model, question, context)
        if max(score) > 0:
            return indexes[i: i + batch_size][score.argmax()]
    return indexes[0]


def get_answer(tokenizer, models, question, texts):
    tfidf, parag_model, answer_model, question_model, yesno_model = models
    n_doc = search_context(model, tokenizer, parag_model, texts, question)
    context = texts[n_doc]
    scores, parags = find_paragraph(tokenizer, parag_model, q, context)
    n = scores.argmax()
    context = parags[n]
    q_type = get_question_type(tokenizer, question_model, question)
    answer = find_answer(tokenizer, answer_model, question, context) if q_type == "SPAN" else find_yesno_answer(tokenizer, yesno_model, question, context)
    return answer, n_doc

def test(n):
    q = data.questions[n]
    title = data.titles[n]
    print(title, q)
    answer, n_doc = get_answer(tokenizer, (model, parag_model, answer_model, question_model, yesno_model), q,
                               data.texts)
    print(data.titles[n_doc])
    print(answer)


if __name__ == "__main__":
    data = pd.read_csv("dataset/data.csv")
    model = load("dataset")
    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    parag_model = BertForNextSentencePrediction.from_pretrained("C:\\Users\\m31k0l2\\Google Диск\\quansw\\primary").eval().cuda()
    answer_model = BertForQuestionAnswering.from_pretrained(
        "C:\\Users\\m31k0l2\\Google Диск\\quansw\\gold").eval()
    question_model = BertForSequenceClassification.from_pretrained(
        "C:\\Users\\m31k0l2\\Google Диск\\quansw\\primary_yesno_questions").eval()
    yesno_model = BertForSequenceClassification.from_pretrained(
        "C:\\Users\\m31k0l2\\Google Диск\\quansw\\primary_yesno").eval()
    n = 0
    q = data.questions[n]
    title = data.titles[n]
    print(title, q)
    answer, n_doc = get_answer(tokenizer, (model, parag_model, answer_model, question_model, yesno_model), q, data.texts)
    print(data.titles[n_doc])
    print(answer)
