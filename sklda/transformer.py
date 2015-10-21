# coding=utf-8

"""
sklda.transformer

"""

import sklda.topics

from sklearn.base import TransformerMixin, BaseEstimator

class LdaTransformer(BaseEstimator, TransformerMixin):
    """
    This Transformer should have a lda model with a gensim tokenizer
    where it consumes a sequence of strings. tokenizes them with a tokenizer
    and maps the topics to an array
    """

    def __init__(self, tokenizer, ldamodel):
        self.tokenizer= tokenizer
        self.ldamodel = ldamodel

    def fit(self):
        pass


    def fit_transform(self, X, y=None, **fit_params):
        topics = self.ldamodel[self.tokenizer.tokenize[X]]
        return sklda.topics.topic2array(topics)
