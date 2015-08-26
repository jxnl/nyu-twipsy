__author__ = 'JasonLiu'

from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.cross_validation import train_test_split

from classification.reporting import ClassificationReporting
from classification.dao import ClassifierAccess


class CustomGridSearch:
    def __init__(self, pipeline, param_grid, n_classes, random, **kwargs):
        self.clf = (RandomizedSearchCV if random else GridSearchCV)(pipeline, param_grid, **kwargs)
        self.n_classes = n_classes

    def set_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=42)
        return self

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)
        print("[GRIDSCORES]")
        print(self.clf.grid_scores_)
        print("[GRIDSCORES]")
        return self

    def generate_report(self):
        self.report = ClassificationReporting(
            self.clf.best_estimator_, self.X_train, self.X_test, self.y_train, self.y_test, self.n_classes
        ).create_report(output=True, show_roc=False)
        self.report["params"] = self.clf.best_params_
        return self



    def write_to_mongo(self):
        ClassifierAccess.write_report(self.report)
        return self

    def __repr__(self):
        return self.clf.__repr__()

    def __str__(self):
        return self.clf.__str__()