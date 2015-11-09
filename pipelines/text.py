from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

from __private import p2, p3, tokenizer, lda100, d1

__author__ = 'JasonLiu'


class Gensim(BaseEstimator, TransformerMixin):
    """
    GensimPhrases
    ~~~~~~~~~~~~~
    """

    def fit(self):
        pass

    def convert2sparse(self, tokens):
        row, col, data = [], [], []
        for doc_id, document in enumerate(lda100[tokens]):
            for topic_id, weight in document:
                row.append(doc_id)
                col.append(topic_id)
                data.append(weight)
        return sparse.csr_matrix((data, (row, col)))

    def transform(self, X, y=None, **fit_params):
        tokens = [
            d1.doc2bow(doc) for doc in p3[p2[X.str.lower().apply(tokenizer.tokenize)]]
            ]
        return self.convert2sparse(tokens)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
