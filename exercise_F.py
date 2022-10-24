import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import loadData


def main():
    print("loading...")
    train_df, ans_df = loadData.load_data_from_smile_csv(
        "data/SM.csv", use_cache=True)
    test_df, test_ans_df = loadData.load_data_from_smile_csv(
        "data/CP.csv", use_cache=True)

    print("fitting...")
    pipeline = Pipeline(
        steps=[('scaler', StandardScaler()), ('model', Lasso())])
    glf = GridSearchCV(pipeline, param_grid={
        "model__alpha": np.arange(0.001, 0.01, 0.0001)}, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    glf.fit(train_df, ans_df)
    print("done")

    print("----------------------------------------")
    test_pred = glf.best_estimator_.predict(test_df)
    print(
        f"テストデータ全体に対するRMSE: {np.sqrt(mean_squared_error(test_ans_df, test_pred))}")
    print(f"相関係数: {np.corrcoef(test_ans_df, test_pred)[0, 1]}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
