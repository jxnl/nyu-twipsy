from re import compile
from functools import lru_cache
from json import load

__author__ = 'JasonLiu'

from pipelines.helpers import ExplodingRecordJoiner
from classification import Taxonomy

ready_made_exploder = ExplodingRecordJoiner(user=[
    'created_at',
    'favourites_count',
    'followers_count',
    'friends_count',
    'statuses_count',
    'verified'
])



# class ReadyMadeHeirarchy:
#     clf_alch = load(open(
#         "./clfs/LogisticRegression|label:alcohol|accuracy:0.7970909090909091|f1:0.8367466354593329|feature:['text', "
#         "'user', 'age']"
#         , "rb+"))
#     clf_frst = load(open(
#         "./clfs/LogisticRegression|label:first_person|accuracy:0.6640826873385013|f1:0.721030042918455|feature:["
#         "'text', 'user', 'age']"
#         , "rb+"))
#     clf_fstl = load(open(
#         "./clfs/LogisticRegression|label:first_person_level|accuracy:0.4703196347031963|f1:0.46356955515952386"
#         "|feature:['text']"
#         , "rb+"
#     ))
#
#     root = Taxonomy(
#         "alcohol",
#         {
#             0: "non_drinking",
#             1: "drinking"
#         },
#         clf_alch["clf"]
#     )
#
#     first_person = Taxonomy(
#         "first_person",
#         {
#             0: "alcohol_related",
#             1: "first_person"
#         },
#         clf_frst["clf"]
#     )
#
#     first_person_level = Taxonomy(
#         "first_person_level",
#         {
#             0: "first_person_casual",
#             1: "first_person_looking",
#             2: "first_person_reflecting",
#             3: "first_person_heavy"
#         },
#         clf_fstl["clf"]
#     )
#
#
#     first_person.add_children({1: first_person_level})
#     root.add_children({1: first_person})

class TurkResults2Label:
    """
    TurkResults2Label
    ~~~~~~~~~~~~~~~~~

    Usage:
        TurkResults2Label.parse_to_labels(string_labe: str)
    """

    first = compile("First")
    alch = compile("Alcohol Consumption")

    drinking_level = {
        "First Person - Alcohol": 0,
        "First Person - Alcohol::Casual Drinking": 0,
        "First Person - Alcohol::Looking to drink": 1,
        "First Person - Alcohol::Reflecting on drinking": 2,
        "First Person - Alcohol::Heavy Drinking": 3
    }

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
