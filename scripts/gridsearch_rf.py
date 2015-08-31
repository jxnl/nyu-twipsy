# coding=utf-8

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

from classification.compute import CustomGridSearch
from pipelines.alcohol import AlcoholPipeline
from data import iterate_heirarchy

from scripts import text_grid

pipeline = AlcoholPipeline(global_features=["text"]).pipeline(RandomForestClassifier())

param_grid = {
    'clf__bootstrap': [True, False],
    'clf__class_weight': ['auto', None],
    'clf__criterion': ['gini'],
    'clf__max_depth': randint(10, 1000),
    'clf__max_features': randint(100, 10000),
    'clf__min_samples_leaf': randint(1, 10),
    'clf__min_samples_split': randint(1, 10),
    'clf__n_estimators': randint(400, 10000),
    'clf__n_jobs': [4],
    'clf__verbose': [0],
}.update(text_grid)

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
    for level, (X, y), n_classes_ in iterate_heirarchy():
        gridsearch = CustomGridSearch(pipeline, param_grid, n_classes_, random=True)
        gridsearch \
            .set_data(X, y) \
            .fit() \
            .generate_report(name="test_batch", level=level, notes="delete") \
            .write_to_mongo()
