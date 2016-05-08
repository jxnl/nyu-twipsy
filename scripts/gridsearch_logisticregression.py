# coding=utf-8

from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

from pipelines.alcohol import AlcoholPipeline
from data import iterate_heirarchy
from classification.compute import CustomGridSearch
from gridsearch import text_grid

pipeline = AlcoholPipeline(global_features=["text", "topic"]).pipeline(LogisticRegression())

param_grid = {
    'clf__C': uniform(0.0001, 1000),
    'clf__penalty': ['l2', "l1"],
    'clf__tol': uniform(0.0001, 0.001),
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

    print(param_grid)

    for level, (X, y), n_classes_ in iterate_heirarchy():
        gridsearch = CustomGridSearch(pipeline, param_grid, n_classes_, random=True, **cv_kwargs)
        gridsearch.set_data(X, y)\
            .fit()\
            .generate_report(
                name="LogisticRegression_LDA_NORM",
                level=level,
                notes="added 1300 + topics")\
            .write_to_mongo()
