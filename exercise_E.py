import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import loadData


def fit_model():
    print("loading...")
    train_df, ans_df = loadData.load_data_from_smile_csv(
        "data/SM.csv", use_cache=True)

    print("fitting...")
    pipeline = Pipeline(
        steps=[('scaler', StandardScaler()), ('model', Lasso())])
    glf = GridSearchCV(pipeline, param_grid={
        "model__alpha": np.arange(0.001, 0.01, 0.0001)}, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    glf.fit(train_df, ans_df)
    print("done")

    print("----------------------------------------")
    print(f"RMSE: {-glf.best_score_}")
    print(f"最良パラメータ: {glf.best_params_}")
    print("----------------------------------------")
    print("coef: ", len(glf.best_estimator_.named_steps["model"].coef_))

    print("----------------------------------------")
    result_df = pd.DataFrame(glf.cv_results_)
    result_df.sort_values(
        by="rank_test_score", inplace=True)
    result_df.to_csv("./output/result.csv")
    print(result_df[["rank_test_score",
                    "params",
                     "mean_test_score"]])

    # 訓練データに対する予測を比較する
    print("----------------------------------------")
    best = glf.best_estimator_
    y_pred = best.predict(train_df)
    print(f"訓練データ全体に対するRMSE: {np.sqrt(mean_squared_error(ans_df, y_pred))}")

    # 相関係数
    print(f"相関係数: {np.corrcoef(ans_df, y_pred)[0, 1]}")
    print("----------------------------------------")


if __name__ == "__main__":
    fit_model()
