__author__ = 'JasonLiu'

from datetime import datetime
import pickle

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


class ClassificationReporting:
    metrics = [accuracy_score, f1_score, confusion_matrix, classification_report]

    def __init__(self, clf, X_train, X_test, y_train, y_test, n_classes):
        """

        :param clf:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param n_classes:
        :return:
        """
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.n_classes = n_classes

        self.report = {}

    def serialize_classifier(self):
        self.report["type"] = type(self.clf.named_steps["clf"]).__name__
        self.report["clf"] = pickle.dumps(self.clf)

    def set_name(self, name):
        """

        :param name:
        :return:
        """
        self.report["name"] = name
        return self

    def set_level(self, level):
        """

        :param level:
        :return:
        """
        self.report["level"] = level
        return self

    def set_notes(self, notes):
        """

        :param notes:
        :return:
        """
        self.report["notes"] = notes
        return self

    def set_params(self, attr_name, attr):
        """

        :param attr_name:
        :param attr:
        :return:
        """
        self.report[attr_name] = attr
        return self

    def set_datetime(self):
        """

        :return:
        """
        self.report["created_at"] = str(datetime.today())
        return self

    def compute_metrics(self, prefix, X, y):
        """

        :param prefix:
        :param X:
        :param y:
        :return:
        """
        y_predict = self.clf.predict(X)
        results = {}
        # simple metrics
        for metric in self.metrics:
            kwargs = {}
            if metric == f1_score:
                kwargs["average"] = "weighted"
            result = metric(y_true=y, y_pred=y_predict, **kwargs)
            results[metric.__name__] = result.tolist() if hasattr(result, "tolist") else result
        self.report[prefix] = results

    def set_path(self):
        name_template = "{level}|accuracy:{accuracy}|f1:{f1_score}|type:{type}"
        self.report["path"] = name_template.format(
            level=self.report["level"],
            accuracy=self.report["testing_results"]["accuracy_score"],
            f1_score=self.report["testing_results"]["f1_score"],
            type=self.report['type']
        )
        return self

    def compute_rocauc(self):
        """

        :return:
        """
        # Binarize the output
        y_test = label_binarize(self.y_test, classes=list(range(self.n_classes)))

        # Compute ROC curve and ROC area for each class
        y_score = self.clf.predict_proba(self.X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        self.report["roc_auc"] = dict(
            fpr={str(k): v.tolist() for k, v in fpr.items()},
            tpr={str(k): v.tolist() for k, v in tpr.items()},
            roc_auc={str(k): v.tolist() for k, v in roc_auc.items()}
        )

    def print(self):
        """

        :return:
        """
        print("Training Results")
        print("~~~~~~~~~~~~~~~~")
        for k, v in self.report["training_results"].items():
            print(k, v, "\n", sep="\n")
        print()
        print()
        print("Testing Results Results")
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        for k, v in self.report["testing_results"].items():
            print(k, v, "\n", sep="\n")
        print()
        print()

    def show(self):
        """

        :return:
        """
        fpr = self.report["roc_auc"]["fpr"]
        tpr = self.report["roc_auc"]["tpr"]
        roc_auc = self.report["roc_auc"]["roc_auc"]

        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

        for i in range(self.n_classes):
            i = str(i)
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for {}'.format(self.report.get("name", "Classifier")))
        plt.legend(loc="lower right")
        plt.show()

    def create_report(self, output=False, show_roc=False):
        """

        :param output:
        :param show_roc:
        :return:
        """
        try:
            self.compute_rocauc()
        except:
            pass
        self.compute_metrics("training_results", self.X_train, self.y_train)
        self.compute_metrics("testing_results", self.X_test, self.y_test)
        self.serialize_classifier()
        self.set_path()

        if output:
            self.print()

        if show_roc:
            self.show()

        return self.report
