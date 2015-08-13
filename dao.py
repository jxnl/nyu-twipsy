import pandas as pd

from config import db

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
    random = project("random_number")
    user = project("friends_count",
                   "followers_count",
                   "statuses_count",
                   "favourites_count",
                   "created_at",
                   "verified",
                   prefix="user")

    all = union(text, predict, time, user, labels)
    mechanical_turk = union(text, predict, random)


class Queries:
    """
    This serves as a collection of different queries we do on the
    collection from MongoDB.
    """
    X = exists("labels")
    no_label = exists("labels", e=False)

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

    @classmethod
    def to_df(cls, cursor):
        return pd.DataFrame(list(cursor)).set_index("_id")

    @classmethod
    def get_as_dataframe(cls, find=Queries.X, projection=Projections.all):
        return cls.to_df(db.find(find, projection))

    @classmethod
    def get_not_labeled(cls):
        return cls.to_df(db.find(Queries.no_label, Projections.mechanical_turk))

    @classmethod
    def write_labels(cls, series):
        for _id,label in series.to_dict().items():
            db.find_one_and_update({"_id":_id}, {"$set": {"labels": label}})

    @classmethod
    def count_withlabels(cls):
        return db.find(exists("labels")).count()
