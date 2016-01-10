# coding=utf-8

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

from classification.compute import CustomGridSearch
from pipelines.alcohol import AlcoholPipeline
from data import iterate_heirarchy
from gridsearch import text_grid

pipeline = AlcoholPipeline(global_features=["text", "topic"]).pipeline(
    RandomForestClassifier()
)

param_grid = {
    'clf__bootstrap': [True, False],
    'clf__criterion': ['gini'],
    'clf__max_depth': randint(10, 1000),
    'clf__max_features': randint(100, 200),
    'clf__min_samples_leaf': randint(1, 10),
    'clf__min_samples_split': randint(1, 10),
    'clf__n_estimators': randint(400, 10000),
    'clf__n_jobs': [4],
    'clf__verbose': [0],
}

param_grid.update(text_grid)

cv_kwargs = dict(
    n_iter=50,
    scoring=None,
    fit_params=None,
    n_jobs=4,
    iid=True,
    refit=True,
    cv=None,
    verbose=3,
    pre_dispatch='2*n_jobs',
    error_score=0
)

if __name__ == "__main__":
    for level, (X, y), n_classes_ in iterate_heirarchy():
        gridsearch = CustomGridSearch(pipeline, param_grid, n_classes_, random=True, **cv_kwargs)
        print(y)
        gridsearch \
            .set_data(X, y) \
            .fit() \
            .generate_report(name="RF_LDA", level=level, notes="added 1300, nov17") \
            .write_to_mongo()
