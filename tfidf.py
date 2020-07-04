import pandas as pd
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.corpora.textcorpus import TextCorpus
import nltk
from nltk.corpus import stopwords
from gensim import utils
from gensim import corpora
import re
from nltk.stem.snowball import SnowballStemmer
from gensim.similarities import MatrixSimilarity
import pymorphy2
from tqdm import tqdm

from gensim.models import Phrases
from gensim.models.phrases import Phraser

nltk.download("stopwords")
rus_stopwords = stopwords.words("russian")
stemmer = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()


def preprocess_text(text):
    s = " ".join([word for word in text.lower().replace('́', '').split() if word not in rus_stopwords])
    words = [s for s in re.findall(f'(\w+)', s)]
    words = [morph.parse(word)[0] for word in words]
    words = [p.normal_form for p in words]
    # words = [stemmer.stem(w) for w in words]
    return words


def index_doc(doc: str, df: pd.DataFrame):
    doc = " ".join(preprocess_text(doc))
    df.append(pd.DataFrame({"doc": [doc]}), ignore_index=True)


def index_docs(df: pd.DataFrame):
    index = []
    for doc in tqdm(df.texts):
        index += [" ".join(preprocess_text(doc))]
    return pd.DataFrame({"docs": index})


def setup_and_save(save_dir):
    index = pd.read_csv(f"{save_dir}/index.csv")
    tokens = [s.split() for s in index.docs.to_list()]
    bigram = Phrases(tokens, min_count=1, threshold=2, delimiter=b' ')
    bigram_phraser = Phraser(bigram)
    bigram_phraser.save(f"{save_dir}/bigram.phr")
    dataset = [bigram_phraser[sent] for sent in tokens]

    dct = Dictionary(dataset)
    dct.save(f"{save_dir}/dictionary.dct")
    corpus = [dct.doc2bow(line) for line in dataset]
    corpora.MmCorpus.serialize(f"{save_dir}/bow_corpus.mm", corpus)
    model = TfidfModel(corpus)
    model.save(f"{save_dir}/tfidf.model")


def load(save_dir):
    dct = corpora.Dictionary.load(f"{save_dir}/dictionary.dct")
    corpus = corpora.MmCorpus(f"{save_dir}/bow_corpus.mm")
    model = TfidfModel.load(f"{save_dir}/tfidf.model")
    index = MatrixSimilarity(corpus, num_features=len(dct))
    bigram_phraser = Phraser.load(f"{save_dir}/bigram.phr")
    return dct, model, index, bigram_phraser


def search(model, query):
    dct, model, index, bigram_phraser = model
    q = preprocess_text(query)
    q = list(set(q + bigram_phraser[q]))
    sims = index[model[dct.doc2bow(q)]]
    return sims.argsort()[::-1], sims


if __name__ == "__main__":
    index = index_docs(pd.read_csv("dataset/data.csv"))
    index.to_csv("dataset/index.csv")

    setup_and_save("dataset")
    print("Модель обучена")

