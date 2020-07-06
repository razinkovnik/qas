from corus import load_wiki
import re
import itertools
import pandas as pd
from tqdm import tqdm


# https://github.com/natasha/corus


if __name__ == "__main__":
    path = 'dataset/ruwiki-latest-pages-articles.xml.bz2'
    records = load_wiki(path)
    data = {
        "title": [],
        "text": []
    }
    for doc in tqdm(records):
        data['title'] += [doc.title]
        data['text'] += [doc.text]
    df = pd.DataFrame(data)
    df.to_csv("dataset/wiki.csv")
