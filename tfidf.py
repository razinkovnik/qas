#%%

import pandas as pd

#%%

df = pd.read_csv("../dataset/docs.csv")

#%%

data = df.docs.tolist()

#%%

import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

#%%

from gensim.corpora.textcorpus import TextCorpus
import nltk
from nltk.corpus import stopwords
from gensim import utils
import re
rus_stopwords = stopwords.words("russian")
# nltk.download("stopwords")
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

def preprocess_text(text):
    s = " ".join([word for word in text.lower().split() if word not in rus_stopwords])
    return [stemmer.stem(w) for w in re.findall(f'(\w+)', s)]

def gen_dataset():
    for doc in data:
        yield preprocess_text(doc)


#%%

dataset = [wd for wd in gen_dataset()]

#%%

dct = Dictionary(dataset)

#%%

corpus = [dct.doc2bow(line) for line in dataset]

#%%

model = TfidfModel(corpus)

#%%

q = df.questions[1040]
print(q)
q = preprocess_text(q)
print(q)
q = model[dct.doc2bow(q)]

#%%

from gensim.similarities import MatrixSimilarity

#%%

index = MatrixSimilarity(corpus, num_features=len(dct))

#%%

sims = index[q]
sorted(enumerate(sims), key=lambda a: a[1], reverse=True)

#%%



