import json
import os

from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LdaModel
from tqdm import tqdm

from src.utils import save
from config import MAIN_EXPERIMENT, LDA_HYPERPARAMS


def main():

    with open(MAIN_EXPERIMENT.preprocessed_file, 'r', encoding='utf-8') as f:
        preprocessed = json.load(f)[:]

    print(f'{len(preprocessed)=}')

    flatten = [par for policy in preprocessed for par in policy]

    dictionary = corpora.Dictionary(flatten)
    corpus = [dictionary.doc2bow(doc) for doc in flatten]
    tfidf = TfidfModel(corpus, id2word=dictionary)
    corpus = tfidf[corpus]
    
    for t in tqdm(MAIN_EXPERIMENT.topics_cnt, ncols=80, ascii=True):
        hyperparameters = LDA_HYPERPARAMS.copy()
        hyperparameters['num_topics'] = t
        
        lda = LdaModel(corpus, id2word=dictionary, **hyperparameters)
        
        save(lda, dictionary, corpus, f'{MAIN_EXPERIMENT.models}/{t}')

        with open(f'{MAIN_EXPERIMENT.topics}/{t}.json', 'w', encoding='utf-8') as f:
            json.dump(
                lda.print_topics(num_topics=hyperparameters['num_topics'], num_words=20), 
                f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
    