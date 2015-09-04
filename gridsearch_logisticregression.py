# coding=utf-8

from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

from pipelines.alcohol import AlcoholPipeline
from data import iterate_heirarchy
from classification.compute import CustomGridSearch
from gridsearch import text_grid

pipeline = AlcoholPipeline(global_features=["text"]).pipeline(LogisticRegression())

param_grid = {
    'clf__C': uniform(0.01, 1000),
    'clf__class_weight': ["auto", None],
    'clf__penalty': ['l2', 'l1'],
    'clf__tol': uniform(0.00001, 0.001),
    'clf__verbose': [0],
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
    verbose=3,
    pre_dispatch='2*n_jobs',
    error_score='raise'
)

if __name__ == "__main__":

    print(param_grid)

    for level, (X, y), n_classes_ in iterate_heirarchy():
        gridsearch = CustomGridSearch(pipeline, param_grid, n_classes_, random=True, **cv_kwargs)
        gridsearch.set_data(X, y)

        print(gridsearch.clf.get_params().keys())

        gridsearch.fit()
        #    .generate_report(name="test_batch", level=level, notes="delete") \
        #    .write_to_mongo()