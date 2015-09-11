__author__ = 'JasonLiu'

from functools import lru_cache

import re

class TurkResults2Label:
    """
    TurkResults2Label
    ~~~~~~~~~~~~~~~~~

    Usage:
        TurkResults2Label.parse_to_labels(string_labe: str)
    """

    first = re.compile("First")
    alch = re.compile("Alcohol Consumption")

    drinking_level = {
        "First Person - Alcohol": 0,
        "First Person - Alcohol::Casual Drinking": 0,
        "First Person - Alcohol::Looking to drink": 1,
        "First Person - Alcohol::Reflecting on drinking": 2,
        "First Person - Alcohol::Heavy Drinking": 0
    }  # Heavy Drinking is collasped into "drinking" since the label is unsuccessful

    related = {
        "Alcohol Related::Discussion": 0,
        "Alcohol Related::Promotional Content": 1
    }

    @classmethod
    @lru_cache(20)
    def parse_to_labels(cls, string_label):
        """
        :param string_label: amazon turk classification result
        :return: dictionary of classifications
        """
        label = {}
        if string_label == "Not Alcohol Related":
            label["alcohol"] = 0
            return label
        else:
            label["alcohol"] = 1

        if cls.alch.match(string_label) and not cls.first.match(string_label):
            return label

        if cls.first.match(string_label):
            label["first_person"] = 1
            label["first_person_level"] = cls.drinking_level[string_label]
            return label
        else:
            label["first_person"] = 0
            label["alcohol_related"] = cls.related[string_label]
        label["raw"] = string_label
        return label
