from operator import itemgetter

import pandas as pd

from __private import db, db2
from data.helpers import ready_made_exploder

__author__ = 'JasonLiu'


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
    control = project("control")
    random = project("random_number")
    user = project(
        "friends_count",
        "followers_count",
        "statuses_count",
        "favourites_count",
        "created_at",
        "verified",
        prefix="user"
    )

    all = union(text, predict, time, user, labels)
    mechanical_turk = union(text, predict, random, control)


class Queries:
    """
    This serves as a collection of different queries we do on the collection from MongoDB.
    """
    X = exists("labels")
    no_label = exists("labels", e=False)

    @classmethod
    def sample(cls, lower=0.0, upper=1.0):
        """
        Each item in the collection has a random number,
        this way we can deterministically sample the collection.
        For all elements:

            `random_number in (lower, upper)`

        :param lower:
        :param upper:
        :return: query object for MongoDB
        """
        return {"random_number":
            {
                "$gt": lower,
                "$lt": upper
            }
        }


class DataAccess:
    @classmethod
    def sample_control(cls, lower=0, upper=.01):
        return cls.to_df(db2.find(find=Queries.sample(lower, upper), projection=Projections.all))

    @classmethod
    def to_df(cls, cursor):
        return pd.DataFrame(list(cursor)).set_index("_id")

    @classmethod
    def get_as_dataframe(cls, find=Queries.X, projection=Projections.all, explode=True):
        df = cls.to_df(db.find(find, projection))
        if explode:
            return ready_made_exploder.fit_transform(df)
        else:
            return df

    @classmethod
    def get_not_labeled(cls):
        return cls.to_df(db.find(Queries.no_label, Projections.mechanical_turk))

    @classmethod
    def write_labels(cls, series):
        for _id, label in series.to_dict().items():
            db.find_one_and_update({"_id": _id}, {"$set": {"labels": label}})

    @classmethod
    def count_withlabels(cls):
        return db.find(exists("labels")).count()


class LabelGetter:
    alcohol = "alcohol"
    first_person = "first_person"
    first_person_level = "first_person_level"

    def __init__(self, X):
        self.X = X

    def get_flatlabels(self):
        labels = self.X["labels"]
        return self.X, labels.apply(self._flatten)

    def get_alcohol(self):
        """
        :return: X, y
        """
        return self._get_labels(self.alcohol)

    def get_first_person(self):
        """
        :return: X, y
        """
        return self._get_labels(self.first_person)

    def get_first_person_label(self):
        """
        :return: X, y
        """
        X, y = self._get_labels(self.first_person_level)
        y[y == 3] = 0 # Fixed the problem of heavy labels
        return X, y

    def _flatten(self, label_dict):
        """
        :param label_dict:
        :return:
        """
        if self.first_person_level in label_dict:
            return label_dict[self.first_person_level] + 2
        else:
            return label_dict[self.alcohol]

    def _get_labels(self, label_name):
        """
        :param label_name:
        :return: X, y with label_name
        """
        labels = self.X["labels"]
        if label_name == "flat":
            return self.get_flatlabels()
        has = labels.apply(self._contains(label_name))
        return self.X[has], labels[has].apply(itemgetter(label_name))

    @classmethod
    def _contains(cls, a):
        def c(b): return a in b

        return c

    def __repr__(self):
        return self.X.__repr__()
