# coding=utf-8

from sklearn.ensemble import RandomForestClassifier

from classification.compute import CustomGridSearch
from pipelines.alcohol import AlcoholPipeline
from data import iterate_heirarchy

pipeline = AlcoholPipeline(global_features=["text"]).pipeline(RandomForestClassifier())

param_grid = {
    'clf__bootstrap': True,
    'clf__class_weight': None,
    'clf__criterion': 'gini',
    'clf__max_depth': None,
    'clf__max_features': 'auto',
    'clf__max_leaf_nodes': None,
    'clf__min_samples_leaf': 1,
    'clf__min_samples_split': 2,
    'clf__min_weight_fraction_leaf': 0.0,
    'clf__n_estimators': 10,
    'clf__n_jobs': 1,
    'clf__oob_score': False,
    'clf__random_state': None,
    'clf__verbose': 0,
    'clf__warm_start': False,
    'features__text__tfidf__analyzer': 'word',
    'features__text__tfidf__binary': False,
    'features__text__tfidf__input': 'content',
    'features__text__tfidf__lowercase': True,
    'features__text__tfidf__max_df': 1.0,
    'features__text__tfidf__max_features': None,
    'features__text__tfidf__min_df': 1,
    'features__text__tfidf__ngram_range': (1, 1),
    'features__text__tfidf__norm': 'l2',
    'features__text__tfidf__preprocessor': None,
    'features__text__tfidf__smooth_idf': True,
    'features__text__tfidf__stop_words': None,
    'features__text__tfidf__strip_accents': None,
    'features__text__tfidf__sublinear_tf': False,
    'features__text__tfidf__tokenizer': None,
    'features__text__tfidf__use_idf': True,
    'features__text__tfidf__vocabulary': None
}

cv_kwargs = dict(
    n_iter=30,
    scoring=None,
    fit_params=None,
    n_jobs=1,
    iid=True,
    refit=True,
    cv=None,
    verbose=0,
    pre_dispatch='2*n_jobs',
    error_score='raise'
)

if __name__ == "__main__":
    for level, (X, y), n_classes_ in iterate_heirarchy()
        gridsearch = CustomGridSearch(pipeline, param_grid, n_classes_, random=True)
        gridsearch \
            .set_data(X, y) \
            .fit() \
            .generate_report(name="test_batch", level=level, notes="delete") \
            .write_to_mongo()
