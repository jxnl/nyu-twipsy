from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

from pipelines.helpers import ItemGetter, ExplodingRecordJoiner
from pipelines.user import UserAgeMonths, UserEgoVectorizer
from pipelines.time import Timestamp2DatetimeIndex, DatetimeIndexAttr

__author__ = 'JasonLiu'


class AlcoholPipeline:
    def __init__(self, time_features=None, lsi=False):
        """
        :param time_features: default(["dayofweek", "hour", "hourofweek"])
        :param lsi: if true, includes the TruncatedSVD() piece
        """
        self.lsi = lsi
        self.time_features = [
            "dayofweek", "hour", "hourofweek"
        ] if not time_features else time_features

    @property
    def _exploder(self):
        return ExplodingRecordJoiner(
            labels=[
                'alcohol'
            ],
            user=[
                'created_at',
                'favourites_count',
                'followers_count',
                'friends_count',
                'statuses_count',
                'verified'
            ]
        )

    def feature_textpipe(self):
        """
        :return: Pipeline(ItemGetter -> TfidfVectorizer -> TruncatedSVD)
        """
        textpipe = [
            ("getter", ItemGetter("text")),
            ("tfidf", TfidfVectorizer()),
        ]
        if self.lsi:
            textpipe.append(("lsi", TruncatedSVD()))
        return Pipeline(textpipe)

    def feature_agepipe(self):
        """
        :return: UserAgeMonths
        """
        agepipe = [
            ("user_age_months", UserAgeMonths())
        ]
        return Pipeline(agepipe)

    def feature_timepipe(self):
        """
        :return: Pipeline(ItemGetter -> Timestamp2DatetimeIndex -> DatetimeIndexAttr)
        """
        timepipe = [
            ("getter", ItemGetter("created_at")),
            ("to_datetimeindex", Timestamp2DatetimeIndex())
        ]

        featureunion = list(
            map(
            lambda datetime_attr: (
                datetime_attr, Pipeline([
                    ("index", DatetimeIndexAttr(datetime_attr)),
                    ("onehot", OneHotEncoder())
                ])
            ), self.time_features)
        )

        timepipe.append(("features", FeatureUnion(featureunion)))
        return Pipeline(timepipe)

    def feature_userpipe(self):
        """
        :return: Pipeline(UserGeoVectorizer)
        """
        userpipe = [
            ("user_ego", UserEgoVectorizer())
        ]
        return Pipeline(userpipe)

    def features(self):
        """
        :return: FeatureUnion([
            ("text", Pipeline(textpipe)),
            ("user", Pipeline(userpipe)),
            ("time", Pipeline(timepipe)),
            ("age", Pipeline(agepipe)),
        ])
        """
        features = [
            ("text", self.feature_textpipe()),
            ("user", self.feature_userpipe()),
            ("time", self.feature_timepipe()),
            ("age", self.feature_agepipe()),
        ]
        return FeatureUnion(features)

    def pipeline(self, clf=None):
        """
        :param clf: sklearn.classifer
        :return: Pipeline([
            ("exploder", exploder),
            ("features", features),
        ])
        """
        pipeline = [
            ("exploder", self._exploder),
            ("features", self.features()),
        ]
        if clf:
            pipeline.append((("clf", clf)))
        return Pipeline(pipeline)
