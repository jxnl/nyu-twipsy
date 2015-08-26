__author__ = 'JasonLiu'

from __private import dbc

import pickle

class ClassifierAccess:
    """
    Provides methids to access the classifiers written to Mongo from training stages
    """
    @classmethod
    def write_report(cls, report):
        return dbc.insert_one(report)

    @classmethod
    def get_best_clf(cls, level="alcohol", metric="test.f1_score"):
        clfs = dbc.find_one({"level": level}, {"clf": 1, "training_results": 1, "testing_results": 1}) \
            .sort(metric, -1).limit(1)
        report = list(clfs)[0]
        report["clf"] = pickle.loads(report["clf"])
