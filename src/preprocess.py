import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymystem3 import Mystem
from tqdm import tqdm

from src.utils import check_files
from config import MAIN_EXPERIMENT


FILES = check_files(MAIN_EXPERIMENT.dataset.documents, r'.*')

TAGS = re.compile(r'\{.*\}')
CHARS = re.compile(r'[^a-zа-яё\-]')
DASH1 = re.compile(r'([^a-zа-яё]|^)-([^a-zа-яё]|$)')
DASH2 = re.compile(r'([a-zа-яё]|^)-([^a-zа-яё]|$)')
DASH3 = re.compile(r'([^a-zа-яё]|^)-([a-zа-яё]|$)')
SPACES = re.compile(r'\s+')
EN = re.compile(r'[a-z]')


def lemmatize(s, t):
    return ''.join(s.lemmatize(t)).replace('\n', '')


def lemmatize_complex(s, t):
    return '-'.join([lemmatize(s, x) for x in t.split('-')])


def preprocess(docs_split, lem):

    stops = list(stopwords.words('russian'))
    stops.extend(stopwords.words('english'))
    stops.extend(MAIN_EXPERIMENT.custom_stopwords)

    clear_docs_split = []
    for doc in tqdm(docs_split, ncols=80, ascii=True):

        pars = []
        for par in doc:
            clean_par = par.lower()
            clean_par = TAGS.sub('', clean_par)
            clean_par = CHARS.sub(' ', clean_par)
            clean_par = DASH1.sub(' ', clean_par)
            clean_par = DASH2.sub('\1 ', clean_par)
            clean_par = DASH3.sub(' \2', clean_par)
            clean_par = SPACES.sub(' ', clean_par)

            tokens = [t for t in clean_par.split(' ') if t]
            tokens = [t for t in tokens if t not in stops]
            tokens = [lemmatize_complex(lem, t) if '-' in t else lemmatize(lem, t) for t in tokens]

            if tokens:
                pars.append(tokens)

        if pars:
            clear_docs_split.append(pars)

    return clear_docs_split


def main():

    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

    print(f'{len(FILES)=}')

    policies = []
    for f in FILES:
        with open(f, 'r', encoding='utf-8') as i:
            policies.append(i.read())
    
    docs_split = [d.lower().split('\n\n') for d in policies]

    lemmatizer = Mystem() if MAIN_EXPERIMENT.preprocess_type == 'ru' else WordNetLemmatizer()
    preprocessed = preprocess(docs_split, lemmatizer)  

    with open(MAIN_EXPERIMENT.preprocessed_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
