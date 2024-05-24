import re
from os import walk
from os.path import join, abspath

from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import MmCorpus
import numpy
import plotly.express as px


def check_files(path, pattern):
    path = abspath(path)
    fs = []
    for dir_path, _, file_names in walk(path):
        fs.extend([join(dir_path, f) for f in file_names if re.match(pattern, f) is not None])
    return fs


def resolve_group_name(id, groups):
    for g in groups:
        if id == g['id']:
            return g['name']
    return None


def resolve_stats(stats):
    if all(s in stats.keys() for s in ['table', 'ol', 'ul', 'li', 'p', 'br']):
        return {
            'length': stats['length'],
            'tables': stats['table'],
            'ordered lists': stats['ol'],
            'unordered lists': stats['ul'],
            'headings': stats['p'],
            'paragraphs': stats['br'],
        }
    return stats


def make_hist(data, groups, threshold):
    min_ = min(data)
    step = (max(data) - min_) / groups
    hist = [round((l - min_) / step) for l in data]
    hist = [hist.count(i) for i in range(groups + 1)]
    hist_squeeze_last = []
    for i, h in enumerate(hist):
        s = sum(hist[i:])
        if s < threshold:
            hist_squeeze_last.append(s)
            break
        else:
            hist_squeeze_last.append(h)
    return hist_squeeze_last, min_, step


def make_hist3(data, groups, step):
    min_ = min(data)
    hist_squeeze_last = [0 for _ in range(groups)]
    for i in range(groups - 1):
         for d in data:
              if min_ + i * step < d and d <= min_ + (i + 1) * step:
                   hist_squeeze_last[i] += 1
    hist_squeeze_last[groups - 1] = sum([1 if min_ + (groups - 1) * step <= d else 0 for d in data])
    return hist_squeeze_last, min_, step


def load(path):
    return (
        LdaModel.load(f'{path}.mdl'), 
        corpora.Dictionary.load(f'{path}.dict'), 
        MmCorpus(f'{path}.freq'))


def save(model, dictionary, corpus, path):
    model.save(f'{path}.mdl')
    dictionary.save(f'{path}.dict')
    MmCorpus.serialize(f'{path}.freq', corpus)


# def print_topics(model, words=10):
#     return model.print_topics(num_topics=model.num_topics, num_words=words)


# def get_document_topics(model, corpus, dictionary, minimum_probability=0.7):
#     return model.get_document_topics(dictionary.doc2bow(corpus), minimum_probability=minimum_probability)


def rgba(c, a):
    return 'rgba({}, {}, {}, {})'.format(*[int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)], a)


def imshow_logscale(img, hoverinfo_z_format:str=':.2e', minor_ticks='auto', **kwargs):
	"""The same as `plotly.express.imshow` but with logarithmic color scale.
	
	Arguments
	---------
	img: array like
		The same as `img` for `plotly.express.imshow`.
	hoverinfo_z_format: str, default `':.2e'`
		A formatting string string for displaying the values in the hover
		boxes for the color scale.
	minor_ticks: `'auto'`, `True` or `False`, default `'auto'`
		If `True`, minor ticks (2,3,4,5,...) are shown, if `False` then
		only major ticks are shown (1,10,100,1000, etc). If `'auto'` then
		the decision is made according to the orders of magnitude spanned
		by the data.
	
	Returns
	-------
	fig: plotly.graph_objects._figure.Figure
		A figure, same as `plotly.express.imshow`.
	"""

	if minor_ticks not in {True,False,'auto'}:
		raise ValueError(f'`minor_ticks` must be True, False or "auto", received {repr(minor_ticks)}. ')
	
	log_data = numpy.log10(img)
	fig = px.imshow(
		img = log_data,
		**kwargs,
	)
	TICKS_VALS = [list(numpy.linspace(10**e,10**(e+1),10)[:-1]) for e in range(-18,12)]
	if minor_ticks == 'auto':
		if numpy.nanmax(log_data) - numpy.nanmin(log_data) > 3:
			minor_ticks = False
	if minor_ticks == False:
		TICKS_VALS = [[_[0]] for _ in TICKS_VALS]
	TICKS_VALS = [_ for l in TICKS_VALS for _ in l]
	ticks_text = [str(_) for _ in TICKS_VALS]
	fig.update_layout(
		coloraxis_colorbar = dict(
			tickvals = [numpy.log10(_) for _ in TICKS_VALS],
			ticktext = ticks_text,
		),
	)
	fig['data'][0].customdata = img
	hover_template = fig['data'][0].hovertemplate.split('<br>')
	labels_for_hover_template = [_.split(': %{')[0] for _ in hover_template]
	fig['data'][0].hovertemplate = f"{labels_for_hover_template[0]}: %{{x}}<br>{labels_for_hover_template[1]}: %{{y}}<br>{labels_for_hover_template[2]}: %{{customdata{hoverinfo_z_format}}}<extra></extra>"
	return fig
