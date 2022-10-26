import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import loadData
import exercise_C
import exercise_D


def main():
    # データの読み込み
    print("loading...")
    X_small_df, y_small_df = loadData.load_data_from_smile_csv(
        "data/SM.csv", use_cache=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_small_df, y_small_df, test_size=0.1, random_state=42)

    # モデルの構築
    print("fitting...")
    pipeline = Pipeline(
        steps=[('scaler', StandardScaler()), ('model', Lasso())])
    glf = GridSearchCV(pipeline, param_grid={
        "model__alpha": np.arange(0.001, 0.01, 0.0001)}, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    glf.fit(X_train, y_train)
    print("done")

    # モデルの評価
    print("----------------------------------------")
    print(f"cross validation中の最良RMSE: {-glf.best_score_}")
    print(f"最良パラメータ: {glf.best_params_}")

    print("----------------------------------------")
    coef_df = pd.DataFrame(
        glf.best_estimator_.named_steps["model"].coef_, index=exercise_C.descriptor_names(), columns=["coef"])
    print("## coef")
    print(coef_df.reindex(coef_df.coef.abs().sort_values(ascending=False).index))

    print("----------------------------------------")
    print("## scores")
    result_df = pd.DataFrame(glf.cv_results_)
    result_df.sort_values(
        by="rank_test_score", inplace=True)
    result_df.to_csv("./output/result.csv")
    print(result_df[["rank_test_score",
                    "params",
                     "mean_test_score"]])

    print("----------------------------------------")
    print("## test")
    best = glf.best_estimator_
    y_pred = best.predict(X_test)
    print(f"テストデータセットに対するRMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"テストデータセットでの相関係数: {np.corrcoef(y_test, y_pred)[0, 1]}")
    plt.scatter(list(map(exercise_D.toPPB, y_pred)),
                list(map(exercise_D.toPPB, y_test)))
    plt.xlabel(r"$Test\, Predicted\, f_b$")
    plt.ylabel(r"$Test\, f_b$")
    plt.savefig("./output/ex_E_test.png")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
