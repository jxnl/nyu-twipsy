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

    def get_reports(cls, level="alcohol", metric="testing_results.f1_score", get_params=0):
        clfs = dbc.find(
            {
                "level": level
            },
            {
                "type": 1,
                "level": 1,
                "clf":1,
                "training_results.accuracy_score": 1,
                "training_results.f1_score": 1,
                "testing_results.accuracy_score": 1,
                "testing_results.f1_score": 1,
            }).sort(metric, -1)
        return list(clfs)

    @classmethod
    def get_byfile(cls, filename):
        return pickle.loads(fs.find_one({"filename": filename}).read())
