import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

__author__ = 'JasonLiu'


class DatetimeIndexAttr(BaseEstimator, TransformerMixin):
    """
    DatetimeIndexAttr
    ~~~~~~~~~~~~~~~~~

    Accesses all of the available `pandas.DatetimeIndex` attributes when initialized.
    Also provides a new attribute called "hourofweek".

    Usage:
        Initialize it with kind=`attribute` that you want, for example `hour` or `dayofweek`
    """

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        n = len(X)
        if self.kind == "hourofweek":
            col = X.dayofweek * 24 + X.hour
        else:
            col = getattr(X, self.kind)
        return pd.DataFrame(col)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class Timestamp2DatetimeIndex(BaseEstimator, TransformerMixin):
    """
    Timestamp2DatetimeIndex
    ~~~~~~~~~~~~~~~~~~~~~~~

    This consumes a timestamp series and applies `pandas.DatetimeIndex`
    to return a DatetimeIndex object
    """

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return pd.DatetimeIndex(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
