import numpy as np
import plotly.express as px

from src.classes import Dataset, Experiment
from src.mapping import GROUPS_RU
from src.utils import rgba


COLORS_A = [rgba(c, 0.30) for c in [*px.colors.qualitative.Dark24[:5], *px.colors.qualitative.Dark24[7:]]]
COLORS_B = [rgba(c, 0.75) for c in [*px.colors.qualitative.Dark24[:5], *px.colors.qualitative.Dark24[7:]]]
COLORS_C = [rgba(c, 1.00) for c in [*px.colors.qualitative.Dark24[:5], *px.colors.qualitative.Dark24[7:]]]

STOPWORDS = [
    'который', 'иной', 'либо', 'свой',
    'технология', 'любой', 'число', 'случай', 'данный', 'нести'
    'ответственность', 'конфиденциальный', 'отношение', 'условие', 
    'осуществляться', 'это', 'весь', 'орган', 'относиться', 'также',
    'ииль', 'далее', 'др', 'г', 'д', 'ип', 'ооо', 'по', 'т', 'нк', 
    'и', 'или', 'кроме', 'какой-либо', 'какой-то', 'из-за',
    'print-label', 'iframe', "list", "item", "href", "removed",
    "hyperref", "i", "me", "my", "elf", "we", "our", "selves", "you",
    "r", "rs", "rself", "rselves", "he", "him", "his", "self", "she",
    "her", "it", "its", "elf", "ir", "irs","mselves", "t", "ch", 
    "who", "se", "am", "is", "are", "was", "be", "n", "ng", "has", 
    "had", "do", "did","ng", "a", "an","the", "and", "but", "if", 
    "or", "ause", "as", "il", "le", "of", "at", "by", "for", "ut", 
    "inst", "ween", "o", "ough", "ore", "ve", "ow", "to", "up", 
    "out", "on", "off", "r", "ther", "re", "in", "er", "why", 
    "how", "all", "any", "few", "e", "h", "no", "nor", "not", 
    "own", "so", "too", "y", "can", "l", "don", "uld", "now", 
    "e", "s", "m", "ing"]

NOTEBOOKS = 'notebooks'
RESOURCES = 'resources'

RU = Dataset(f'{RESOURCES}/datasets/ru', 'output_policies', 'output.json', 'metrics.json')
EN = Dataset(f'{RESOURCES}/datasets/en', 'plain_policies', 'plain.json', 'metrics.json')


LDA_HYPERPARAMS = {
    'minimum_probability': 0.01,
    'minimum_phi_value': 0.01,
    'gamma_threshold': 0.001, 
    'per_word_topics': False,
    'distributed': False, 
    'alpha': 'symmetric',
    'dtype': np.float32,
    'callbacks': None, 
    'random_state': 0,
    'update_every': 0, 
    'chunksize': 2000, 
    'num_topics': 100, 
    'eval_every': 10,
    'iterations': 50,
    'ns_conf': None,
    'offset': 1.0, 
    'decay': 0.5,
    'passes': 5,
    'eta': None}

KMEANS_HYPERPARAMS = {
    'random_state': 42,
    'n_clusters': 50,
    'init': 'random',
    'n_init': 'auto',
    'max_iter': 500}


NOTEBOOKS_ARGS_DEFAULT = {
    'notebooks/cluster_bars.ipynb': 
        35,
    'notebooks/cluster_by_1_kmeans.ipynb': 
        [300, 35],
    'notebooks/parse_topics.ipynb': 
        [7, 7, 0.03, 0.08, 4800, 3000],
    'notebooks/pyvis.ipynb': 
        .009,
    'notebooks/structure_5_histograms.ipynb': 
        [(250, 50), (200, 50), (200, 25), (150, 50), (50, 25)],
    'notebooks/structure_by_1_kmeans_wo_hp.ipynb': 
        300,
    'notebooks/structure_by_1_kmeans.ipynb': 
        300,
    'notebooks/structure_heatmap.ipynb': 
        [],
    'notebooks/structure_sizes.ipynb': 
        [(250, 50), (6450, 7500)]}


"""
Experiment with model in Russian 
"""
ARGS_RU1 = NOTEBOOKS_ARGS_DEFAULT.copy()
EX_RU1 = Experiment(
    kmeans_hyperparams=KMEANS_HYPERPARAMS.copy(),
    lda_hyperparams=LDA_HYPERPARAMS.copy(),
    custom_stopwords=STOPWORDS.copy(),
    nb_args=ARGS_RU1.copy(),
    experiment=f'{RESOURCES}/ru1',
    preprocess_type='ru',
    groups=GROUPS_RU,
    dataset=RU,)


"""
Experiment with model in English 
"""
ARGS_EN1 = NOTEBOOKS_ARGS_DEFAULT.copy()
ARGS_EN1['parse_topics.ipynb'] = (6, 4, 0.03, 0.08, 4800, 3000)
EX_EN1 = Experiment(
    kmeans_hyperparams=KMEANS_HYPERPARAMS.copy(),
    lda_hyperparams=LDA_HYPERPARAMS.copy(),
    custom_stopwords=STOPWORDS.copy(),
    nb_args=ARGS_EN1,
    experiment=f'{RESOURCES}/en1',
    preprocess_type='en',
    dataset=EN,)


"""
Main experiment
"""
MAIN_EXPERIMENT = EX_EN1
