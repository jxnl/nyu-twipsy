import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

__author__ = 'JasonLiu'


class ItemGetter(BaseEstimator, TransformerMixin):
    """
    ItemGetter
    ~~~~~~~~~~

    ItemGetter is a Transformer for Pipeline objects.

    Usage:
        Initialize the ItemGetter with a `key` and its
        transform call will select a column out of the
        specified DataFrame.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return X[self.key]

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class ExplodingRecordJoiner(BaseEstimator, TransformerMixin):
    """
    ExplodingRecordJoiner
    ~~~~~~~~~~~~~~~~~~~~~

    ExplodingRecordJoiner is a Transformer for Pipeline Objects

    Usage:
        The reason we use this is because of the fact that
        using DataFrams is better than using JSON parsing.

        However, the data coming in is nested JSON so this exploder
        allows use to select a `col` that is one level nested dictionary
        (taken from json) and selects the `subcol` and joins
        it to the original DataFrame.
    """

    def __init__(self, **kwargs):
        self.cols = kwargs

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        # Extract column of dicts then apply from_records,
        # Match indicies then select the `subcols` we want,
        # Join with existing DataFrame.
        for col, subcol in self.cols.items():
            new_cols = ["{}.{}".format(col, c) for c in subcol]
            sub = pd.DataFrame.from_records(X[col], index=X.index)[subcol]
            sub.columns = new_cols
            X = X.join(sub)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def __repr__(self):
        st = [k + "=" + str(v) for k, v in self.cols.items()]
        return "ExplodingRecordJoiner({})".format(", ".join(st))
