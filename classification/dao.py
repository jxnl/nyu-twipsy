__author__ = 'JasonLiu'

from __private import dbc, fs

import pickle


class ClassifierAccess:
    """
    Provides methids to access the classifiers written to Mongo from training stages
    """
    @classmethod
    def write_report(cls, report):
        """
        writes the report to GridFS and Classifiers Collections

        this method will extract the model out and write the pickled
        model into GridFS and then write the id into the report and
        write the report into Classifiers

        :param report:
        :return:
        """
        with fs.new_file(
            filename=report["name"]
        ) as fp:
            fp.write(report["clf"])
            report["clf"] = fp._id
        return dbc.insert_one(report)

    @classmethod
    def get_best_clf(cls, level="alcohol", metric="test.f1_score"):
        clfs = dbc.find_one({"level": level}, {"clf": 1, "training_results": 1, "testing_results": 1}) \
            .sort(metric, -1).limit(1)
        report = list(clfs)[0]
        report["clf"] = pickle.loads(report["clf"])
