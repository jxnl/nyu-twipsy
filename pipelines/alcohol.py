from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer

from pipelines.helpers import ItemGetter, ExplodingRecordJoiner
from pipelines.user import UserAgeMonths, UserEgoVectorizer
from pipelines.time import Timestamp2DatetimeIndex, DatetimeIndexAttr
from pipelines.text import Gensim

__author__ = 'JasonLiu'


class AlcoholPipeline:
    def __init__(self, time_features=None, global_features=None, lsi=False, lsi_n=1000):
        """
        :param time_features: default(["dayofweek", "hour", "hourofweek"])
        :param global_features: default(["text", "time", "user", "age"])
        :param lsi: if true, includes the TruncatedSVD() piece
        """
        self.lsi = lsi
        self.lsi_n = lsi_n
        self.time_features = [
            "dayofweek", "hour", "hourofweek"
        ] if not time_features else time_features
        self.global_features = {
            "text", "time", "user", "age"
        } if not global_features else global_features

    @property
    def _exploder(self):
        return ExplodingRecordJoiner(
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
        :return: Pipeline(ItemGetter -> TfidfVectorizer)
        """
        textpipe = [
            ("getter", ItemGetter("text")),
            ("tfidf", TfidfVectorizer()),
        ]
        if self.lsi:
            textpipe.append(("lsi", TruncatedSVD(n_components=self.lsi_n)))
        return Pipeline(textpipe)

    def feature_topicpipe(self):
        """
        :return: Pipeline(ItemGetter -> LDA)
        """
        textpipe = [
            ("getter", ItemGetter("text")),
            ("topics", Gensim()),
        ]
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
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
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

        dd = {
            "text": self.feature_textpipe(),
            "topic": self.feature_topicpipe(),
            "user": self.feature_userpipe(),
            "time": self.feature_timepipe(),
            "age": self.feature_agepipe()
        }

        features = list(map(lambda feature: (feature, dd[feature]), self.global_features))
        return FeatureUnion(features)

    def pipeline(self, clf):
        """
        :param clf: sklearn.classifer
        :return: Pipeline([
            ("exploder", exploder),
            ("features", features),
        ])
        """
        pipeline = [
            #("exploder", self._exploder),
            ("features", self.features()),
            ("scaler", Normalizer()),
            ("clf", clf)
        ]
        return Pipeline(pipeline)
