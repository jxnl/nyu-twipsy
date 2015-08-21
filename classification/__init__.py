import pandas as pd

__author__ = 'JasonLiu'

class Taxonomy:
    """
    Taxonomy
    ~~~~~~~~

    This class is used to compose a modular Classification Taxonomy

    :usage:
        root = TaxonomyNode(
            "alcohol",
            {
                0:"non_drinking",
                1:"drinking"
            },
            alcohol_classifier
        )

        first_person = TaxonomyNode(
            "first_person",
            {
                0:"alcohol_related",
                1:"first_person"
            },
            first_person_classifier
        )

        first_person_level = TaxonomyNode(
            "first_person_level",
            {
                0:"first_person_casual",
                1:"first_person_looking",
                2:"first_person_reflecting",
                3:"first_person_heavy"
            },
            first_person_level_classifier
        )

        first_person.add_children({1: first_person_level})
        root.add_children({1: first_person})
        y_predict = root.predict(X, deep=True)
    """

    def __init__(self, name, label2name, sklearn_classifier):
        """
        :param name:  (str) name of the node/classifier
        :param label2name: (dict[int, str]) maps the label to the name
        :param sklearn_classifier: sklearn classifier hat implements predict
        """
        assert(hasattr(sklearn_classifier, "predict"))
        self.name = name
        self.label2name = label2name
        self.clf = sklearn_classifier
        self.children = None


    def add_children(self, label2node):
        """
        :param label2node: (dict[int, Taxonomy])
        """
        self.children = label2node

    def _name(self):
        def get(label):
            return (label, self.label2name[label]) if label in self.label2name else label
        return get

    def predict(self, X, deep=False):

        # Shallow prediction means that we only want to predict for a single node
        if not deep:
            return pd.Series(self.clf.predict(X), index=X.index)

        # We want to predict for all of the children below
        if deep:
            current_labels = self.predict(X)
            
            print(current_labels.value_counts())

            # If there are children, depth first traverse the taxonomy and replace labels
            # with new classifications in the form (label:int, label:str)

            if self.children:
                new_labels = []

                # Satisfy all the childen nodes
                for (label, child) in self.children.items():
                    # Select the relevant data
                    # Predict new labels and add to new_labels
                    relevant_slice = X[current_labels == label]
                    predicted_slice = child.predict(relevant_slice, deep=True)# .apply(self._name)
                    new_labels.append(predicted_slice.apply(child._name()))

                # Satisfy all the leaf nodes
                for label, _ in self.label2name.items():
                    if not label in self.children:
                        relevant_labels = current_labels[current_labels == label]
                        new_labels.append(relevant_labels.apply(self._name()))
                return pd.concat(new_labels).apply(self._name())
            else:
                return current_labels