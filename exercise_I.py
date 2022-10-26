import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import loadData
import pickle
import exercise_D


def main():
    # データの読み込み
    print("loading...")
    X_train, y_train = loadData.load_data_from_smile_csv(
        "data/SM.csv", use_cache=True)
    X_test, y_test = loadData.load_data_from_smile_csv(
        "data/CP.csv", use_cache=True)

    # モデルの構築
    print("fitting...")
    param_grid = {'model__max_depth': np.arange(10, 15, 1),
                  'model__min_samples_leaf': np.arange(1, 5, 1)}
    pipeline = Pipeline(
        steps=[('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=500))])
    glf = GridSearchCV(pipeline, param_grid, cv=5,
                       scoring="neg_root_mean_squared_error", n_jobs=-1)
    glf.fit(X_train, y_train)

    with open("output/glf.pkl", "wb") as f:
        pickle.dump(glf, f)
    print("done")

    # モデルの評価
    print("----------------------------------------")
    print(f"RMSE: {-glf.best_score_}")
    print(f"最良パラメータ: {glf.best_params_}")

    result_df = pd.DataFrame(glf.cv_results_)
    result_df.sort_values(
        by="rank_test_score", inplace=True)
    result_df.to_csv("./output/result_I.csv")
    print(result_df[["rank_test_score",
                    "params",
                     "mean_test_score"]])

    print("----------------------------------------")
    y_pred = glf.best_estimator_.predict(X_test)
    print(
        f"CPでのRMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"CPとの相関係数: {np.corrcoef(y_test, y_pred)[0, 1]}")
    plt.scatter(list(map(exercise_D.toPPB, y_pred)),
                list(map(exercise_D.toPPB, y_test)))
    plt.xlabel(r"$Test\, Predicted\, f_b$")
    plt.ylabel(r"$Test\, f_b$")
    plt.savefig("./output/ex_I_test.png")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
