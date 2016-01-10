# coding=utf-8

from sklearn.svm import LinearSVC
from scipy.stats import uniform

from pipelines.alcohol import AlcoholPipeline
from data import iterate_heirarchy
from classification.compute import CustomGridSearch
from gridsearch import text_grid

pipeline = AlcoholPipeline(global_features=["text", "topic"]).pipeline((LinearSVC()))

param_grid = {
    'clf__C': uniform(0.01, 1000),
    'clf__class_weight': [None],
    'clf__max_iter': [1000],
    'clf__multi_class': ['ovr'],
    'clf__penalty': ['l2'],
    'clf__tol': uniform(0.0001, 0.01),
    'clf__verbose': [0],
}

param_grid.update(text_grid)

cv_kwargs = dict(
    n_iter=30,
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
        gridsearch \
            .set_data(X, y) \
            .fit() \
            .generate_report(name="LinearSVC", level=level, notes="") \
            .write_to_mongo()
