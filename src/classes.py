import json
import os


class Dataset:
    def __init__(self, path, documents, descriptor, metrics) -> None:
        self.path = path
        self.documents = f'{path}/{documents}'
        self.descriptor = f'{path}/{descriptor}'
        self.metrics = f'{path}/{metrics}'


class Experiment:
    def __init__(self, dataset=None, main_model=None, experiment='some',
                 lda_hyperparams={}, kmeans_hyperparams={}, custom_stopwords=[], 
                 topics_cnt=range(10, 71), affiliation_theshold=.5, excerpts_len=200, 
                 groups=None, preprocess_type='ru', nb_args={}) -> None:
        self.experiment = experiment
        self.dataset = dataset

        self._groups = groups
        self._main_model = main_model

        self.lda_hyperparams = lda_hyperparams
        self.kmeans_hyperparams = kmeans_hyperparams

        self.affiliation_theshold = affiliation_theshold
        self.preprocess_type = preprocess_type
        self.excerpts_len = excerpts_len
        self.nb_args = nb_args

        self.custom_stopwords = custom_stopwords
        self.topics_cnt = topics_cnt

        self.models = f'{experiment}/models'
        self.pictures = f'{experiment}/pictures'
        self.topics = f'{experiment}/topics'

        self.preprocessed_file = f'{experiment}/preprocessed.json'
        self.coherence_file = f'{experiment}/coherence.json'
        self.excerpts_file = f'{experiment}/excerpts.json'
        self.lda_file = f'{experiment}/lda.json'

        self.tsne = f'{experiment}/tsne.npy'

        os.makedirs(self.models, exist_ok=True)
        os.makedirs(self.pictures, exist_ok=True)
        os.makedirs(self.topics, exist_ok=True)

    def _best_model(self):
        with open(self.coherence_file) as f:
            c = json.load(f)
        i = 0
        max_ = c[0][1]
        for (c_i, s, _) in c:
            if s > max_:
                max_ = s
                i = c_i
        return i
    
    def _groups_dummy(self):
        return [{'id': i, 'name': f'Тематика {i}', 'topics': [i]} for i in range(self._best_model())]

    @property
    def main_model(self):
        return self._main_model if self._main_model else f'{self.models}/{self._best_model()}'

    @property
    def topics_file(self):
        return f'{self.topics}/{self._main_model}.json' if self._main_model else f'{self.topics}/{self._best_model()}.json'

    @property
    def groups(self):
        return self._groups if self._groups else self._groups_dummy()
