__author__ = 'JasonLiu'

import pandas as pd

from config import db


def union(*dtts):
    result = {}
    for d in dtts:
        result.update(d)
    return result


def project(*attributes, prefix=None):
    temp = prefix + ".{}" if prefix else "{}"
    return {temp.format(attr): 1 for attr in attributes}


def exists(label, e=True):
    return {label: {"$exists": e}}


class Projections:
    """
    This services as the initial feature set used to project data
    from MongoDB, the code there should also serve as a reference
    """
    text = project("text")
    predict = project("predict")
    labels = project("labels")
    time = project("created_at")
    user = project("friends_count",
                   "followers_count",
                   "statuses_count",
                   "favourites_count",
                   "created_at",
                   "verified",
                   prefix="user")

    all = union(text, predict, time, user, labels)


class Queries:
    """
    This serves as a collection of different queries we do on the
    collection from MongoDB.
    """
    X = exists("labels")

    @classmethod
    def sample(cls, lower=0.0, upper=1.0):
        """
        Each item in the collection has a random number, this way we can deterministically sample
        the collection. for all elemebts where:

            `random_number in (lower, upper)`

        :param lower:
        :param upper:
        :return: query object for MongoDB
        """
        return {"random_number": {"$gt": lower, "$lt": upper}}


class DataAccess:
    X = None

    @classmethod
    def to_df(cls, cursor):
        return pd.DataFrame(list(cursor)).set_index("_id")

    @classmethod
    def as_dataframe(cls):
        if cls.X is None:
            cls.X = cls.to_df(db.find(Queries.X, Projections.all))
        return cls.X
