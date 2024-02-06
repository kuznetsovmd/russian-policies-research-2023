import json

from tqdm import tqdm
from gensim.models import CoherenceModel

from src.utils import load
from config import MAIN_EXPERIMENT


def main():
    with open(MAIN_EXPERIMENT.preprocessed_file, 'r', encoding='utf-8') as f:
        preprocessed = json.load(f)

    print(f'{len(preprocessed)=}')

    flatten = [par for policy in preprocessed for par in policy]

    coherence = []
    for r_i in tqdm(MAIN_EXPERIMENT.topics_cnt, ncols=80, ascii=True):
        lda, dictionary, corpus = load(f'{MAIN_EXPERIMENT.models}/{r_i}')
        coherence.append((
            r_i, CoherenceModel(
                dictionary=dictionary,
                coherence='c_v',
                texts=flatten, 
                model=lda).get_coherence(), 
            lda.log_perplexity(corpus)))

    with open(MAIN_EXPERIMENT.coherence_file, 'w', encoding='utf-8') as f:
        json.dump(coherence, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
