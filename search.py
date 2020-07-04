from tfidf import load, search
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForQuestionAnswering
import pandas as pd
from find_answer import context_score, find_paragraph, find_answer
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


def get_answer(tokenizer, models, question):
    tfidf, parag_model, answer_model = models
    n_doc = search_context(model, tokenizer, parag_model, df.texts, question)
    context = df.texts[n_doc]
    scores, parags = find_paragraph(tokenizer, parag_model, q, context)
    n = scores.argmax()
    context = parags[n]
    answer = find_answer(tokenizer, answer_model, question, context)
    return answer, n_doc


if __name__ == "__main__":

    model = load("dataset")
    df = pd.read_csv("dataset/data.csv")
    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    parag_model = BertForNextSentencePrediction.from_pretrained("C:\\Users\\m31k0l2\\Google Диск\\quansw\\primary").eval().cuda()
    answer_model = BertForQuestionAnswering.from_pretrained(
        "C:\\Users\\m31k0l2\\Google Диск\\quansw").eval()

    q = df.questions[3]
    print(q)
    answer, n_doc = get_answer(tokenizer, (model, parag_model, answer_model), q)
    print(df.titles[n_doc])
    print(answer)