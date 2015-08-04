__author__ = 'JasonLiu'

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
    user = project("friends_count", "followers_count", "statuses_count",
                   "favourites_count", "created_at", prefix="user")

    all = union(text, predict, time, user)


class Queries:
    """
    This serves as a collection of different queries we do on the
    collection from MongoDB.
    """
    X = exists("labels")

    @classmethod
    def sample(cls, lower=0.0, upper=1.0):
        return {"random_number": {"$gt": lower, "$lt": upper}}

    @classmethod
    def X_train(cls):
        return union(cls.X, cls.sample(0.0, 0.7))

    @classmethod
    def X_train(cls):
        return union(cls.X, cls.sample(0.0, 0.7))


class DataAccess:

    @classmethod
    def get_data(cls):
        """

        :return:
        """
        X = db.find(Queries.X, Projections.all)
        y = db.find(Queries.X, Projections.labels)
        return X, y

