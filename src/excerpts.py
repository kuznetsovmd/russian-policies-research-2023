import json

from gensim.models import TfidfModel
from tqdm import tqdm

from src.utils import load
from config import MAIN_EXPERIMENT


def main():
    with open(MAIN_EXPERIMENT.preprocessed_file, 'r') as f:
        preprocessed = json.load(f)[:MAIN_EXPERIMENT.excerpts_len]

    lda, dictionary, corpus = load(MAIN_EXPERIMENT.main_model)
    tfidf = TfidfModel(corpus, id2word=dictionary)

    excerpts = []
    for policy in tqdm(preprocessed):
        for paragraph in policy:
            topics = [t for t, s in lda[tfidf[dictionary.doc2bow(paragraph)]] if s > MAIN_EXPERIMENT.affiliation_theshold]
            excerpts.append((topics, paragraph))

    with open(MAIN_EXPERIMENT.excerpts_file, 'w', encoding='utf-8') as f:
        json.dump(excerpts, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
    