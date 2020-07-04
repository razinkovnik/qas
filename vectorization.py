from typing import BinaryIO
from transformers import BertTokenizer, BertModel
from primary_task import read_data
import torch
import numpy as np
import math
from tqdm import tqdm
import pandas as pd


def setup_and_save(datafile_name, save_dir):
    df = pd.read_csv(datafile_name)
    docs = []
    indexes = []
    for i, text in enumerate(df.texts.to_list()):
        parts = [part for part in text.strip().split('\n') if len(part) > 50]
        indexes += [i] * len(parts)
        docs += parts
    batch_size = 32
    texts = [[doc for doc in docs[i*batch_size:(i+1)*batch_size]] for i in range(math.ceil(len(docs)/batch_size))]

    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
    model.eval()
    model.cuda()
    vectors = []
    for batch in tqdm(texts):
        batch = tokenizer.batch_encode_plus(batch, max_length=32, truncation=True, pad_to_max_length=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**batch)[0]
        out = out.cpu().numpy()
        vectors.append(np.average(out, axis=1))
    vectors = np.vstack(vectors)
    np.save('dataset/vectors.npy', vectors)
    np.save('dataset/indexes.npy', np.array(indexes))



def test(test_n: int) -> int:
    query = queries[test_n]
    with torch.no_grad():
        out = model(input_ids=tokenizer.encode(query, return_tensors="pt", max_length=32, truncation=True).cuda())[0]
    query_vector = out.cpu().numpy()
    query_vector = np.average(query_vector, axis=1)
    dist = np.dot(vectors, query_vector.T).flatten()
    predicts = dist.argsort()[::-1]
    predicts = list(dict.fromkeys([indexes[p] for p in predicts]))
    return predicts.index(test_n)

