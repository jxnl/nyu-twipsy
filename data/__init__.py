__author__ = 'JasonLiu'

from data.dao import DataAccess, LabelGetter


def iterate_heirarchy():
    """
    :yields: (label_name, (X,y), n_classes)
    """
    XX = DataAccess.get_as_dataframe()
    LL = LabelGetter(XX)
    yield "alcohol", LL.get_alcohol(), 2
    yield "first_person", LL.get_first_person(), 2
    yield "first_person_label", LL.get_first_person_label(), 3
