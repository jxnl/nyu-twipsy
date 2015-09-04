# coding=utf-8

from sklearn.svm import SVC
from scipy.stats import uniform

from classification.compute import CustomGridSearch
from pipelines.alcohol import AlcoholPipeline
from data import iterate_heirarchy
from gridsearch import text_grid

pipeline = AlcoholPipeline(global_features=["text"]).pipeline(SVC())

param_grid = {
    'clf__C': uniform(10 ** -4, 10 ** 4),
    'clf__cache_size': [500],
    'clf__class_weight': ['auto', None],
    'clf__coef0': uniform(0, 1),
    'clf__degree': [1, 2, 3],
    'clf__gamma': ['auto'],
    'clf__kernel': ['poly', 'rbf'],
    'clf__probability': [False],
    'clf__tol': uniform(0.0001, 0.001)
}

param_grid.update(text_grid)

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
        gridsearch = CustomGridSearch(pipeline, param_grid, n_classes_, random=True, **cv_kwargs)
        gridsearch \
            .set_data(X, y) \
            .fit() \
            .generate_report(name="test_batch", level=level, notes="delete") \
            .write_to_mongo()
