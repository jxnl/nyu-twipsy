from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Phrases

__author__ = 'JasonLiu'


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Tokenizer
    ~~~~~~~~~

    Usage:
        Initialize with a tokenizer and it will apply the tokenizer to every
        element in the series
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return X.apply(self.tokenizer)


class GensimPhrases(BaseEstimator, TransformerMixin):
    """
    GensimPhrases
    ~~~~~~~~~~~~~

    Usage:
        Initialize with the path to a phrase object
    """

    def __init__(self, phrasepath):
        self.phrase = Phrases.load(phrasepath)

    def fit(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return [phrases for phrases in self.phrase[X]]

