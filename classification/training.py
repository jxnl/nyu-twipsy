"""
No usages, maybe delete this
"""

__author__ = 'JasonLiu'

import pickle

from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from dao import LabelGetter, DataAccess
from pipelines.alcohol import AlcoholPipeline

XX = DataAccess.get_as_dataframe()


class CustomCrossValidationSerializer:
    """Run Crossvalidation and then serialize report along with the best estimator"""

    name_template = "{clf_type}|label:{label}|accuracy:{accuracy}|f1:{f1_score}|feature:{features}"

    def __init__(self, base_clf_class, featurelist, params_distribution, **cv_kwargs):
        """

        :param clf:
        :param params_distribution:
        :param cv_kwargs:
        :param n_iter:
        :param verbose:
        :param
        :return:
        """
        self.params_distribution = params_distribution
        self.base_clf_class = base_clf_class
        self.cv_kwargs = cv_kwargs
        self.featurelist = featurelist

    def set_data(self, X):
        """

        :param X:
        :return:
        """
        self.X = X
        return self

    def set_writepath(self, path):
        """

        :param path:
        :return:
        """
        self.path = path
        return self

    def get_data(self, label):
        """

        :param label:
        :return:
        """
        return LabelGetter(self.X)._get_labels(label)

    def generate_report(self, clf, y_train, y_test, X_train, X_test):
        """

        :param clf:
        :param y_train:
        :param y_test:
        :param X_train:
        :param X_test:
        """
        y_predict = clf.predict(X_test)
        y_predict_training = clf.predict(X_train)

        results = \
            {
                "training_accuracy": metrics.accuracy_score(y_train, y_predict_training),
                "training_f1_score": metrics.f1_score(y_train, y_predict_training, average="weighted"),
                "test_accuracy": metrics.accuracy_score(y_test, y_predict),
                "test_f1_score": metrics.f1_score(y_test, y_predict, average="weighted"),
            }
        if True:
            print("classification_report\n", metrics.classification_report(y_test, y_predict))
            print("confusion_matrics\n", metrics.confusion_matrix(y_test, y_predict))
            print("accuracy_score\n", metrics.accuracy_score(y_test, y_predict))
        return results

    def split_fit_evaluate(self, cv_pipeline, label):
        """Run Gridsearch and return a dictionary of report metrics and the best classifier

        :param cv_pipeline:
        :param label:
        :return: (results: dict, cv_pipeline.best_estimator_)
        """
        assert isinstance(cv_pipeline, RandomizedSearchCV)

        X, y = self.get_data(label)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # crossvalidating and fit
        cv_pipeline.fit(X_train, y_train)
        results = self.generate_report(cv_pipeline, y_train, y_test, X_train, X_test)
        return (results, cv_pipeline.best_estimator_)

    def get_gridsearchcv_with_features(self, features):
        """Generat

        :param features: (list[str])
        :param label: (str)
        :return: results dictionary
        """

        clf = AlcoholPipeline(global_features=features).pipeline(self.base_clf_class()),
        cv_pipeline = RandomizedSearchCV(clf, self.params_distribution, **self.cv_kwargs)
        return cv_pipeline

    def serialize_results(self, results):
        """ Seraliz

        :param results:
        :return:
        """
        path = "./clfs/" if not hasattr(self, "path") else self.path
        name = self.name_template.format(**results)
        with open(path + name, "wb+") as f:
            pickle.dump(results, f)
            return True

    def run(self, features, label):
        """Run gridsearch using the appropriate features on the appropriate label

        :param features:
        :param label:
        :return: results
        """

        cv_pipeline = self.get_gridsearchcv_with_features(features)
        results, best_estimator = self.split_fit_evaluate(cv_pipeline, label)

        results["features"] = features
        results["label"] = label

        results["clf_type"] = self.base_clf_class.__name__
        results["best_params"] = best_estimator.best_params_
        results["clf"] = best_estimator

        self.serialize_results(results)
        return results
