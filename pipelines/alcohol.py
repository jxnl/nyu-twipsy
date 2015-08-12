from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from pipelines.helpers import ItemGetter, ExplodingRecordJoiner
from pipelines.user import UserAgeMonths

__author__ = 'JasonLiu'


class AlcoholPipeline:
    def __init__(self, lsi=False):
        self.explode = self._exploder()
        self.lsi = lsi

    def _exploder(self):
        return ExplodingRecordJoiner(
            labels=[
                "alcohol"
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

    def _add_if_params(self, pipe, name, transformer, params):
        if params:
            pipe.append((name, transformer(**params)))
        else:
            pipe.append((name, transformer()))

    def _textpipe(self, tfidf_params=None, lsi_params=None):
        textpipe = [
            ("getter", ItemGetter("text")),
        ]

        self._add_if_params(textpipe, "tfidf", TfidfVectorizer, tfidf_params)

        if self.lsi:
            self._add_if_params(textpipe, "lsi", TruncatedSVD, lsi_params)

        return textpipe

    def _agepipe(self):
        agepipe = [
            ("user_months", UserAgeMonths())
        ]
        return agepipe



features = FeatureUnion([
    ("text", Pipeline(textpipe)),
    ("user", Pipeline(userpipe)),
    ("time", Pipeline(timepipe))
    ("age", Pipeline(agepipe)),
])

pipeline = [
    ("exploder", exploder)
    ("features", features)
]

Pipeline(
)
