import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

__author__ = 'JasonLiu'


class UserEgoVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, log=True, mean=True):
        self.log = log
        self.mean = mean

        self.features = [
            'user.favourites_count',
            'user.followers_count',
            'user.friends_count',
            'user.statuses_count',
            'user.verified'
        ]

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        U = X[self.features].copy()
        U["user.normality"] = U["user.friends_count"] \
                              / ((U["user.followers_count"] + U["user.friends_count"]) + 1)

        # all features omitting user.verified
        for feature in self.features[:-1]:
            # Adding one fixes the log(0) problem
            U[feature] = np.log(U[feature] + 1)

        if self.mean:
            for feature in self.features[:-1]:
                U[feature + "_mean"] = U[feature] - np.mean(U[feature])
                U[feature + "_std"] = (U[feature] - np.mean(U[feature])) ** 2
        return U.astype(float)

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)


class UserAgeMonths(BaseEstimator, TransformerMixin):
    """
    UserAgeMonths
    ~~~~~~~~~~~~~

    Calculates difference in months between user creation time and tweet creation
    """

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        tweet_time = pd.to_datetime(X["created_at"])
        user_time = pd.to_datetime(X["user.created_at"])
        age = ((tweet_time - user_time).apply(int) // 2.62974e15)
        return np.matrix(age.values).T

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)
