import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import loadData
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
    pipeline = Pipeline(
        steps=[('scaler', StandardScaler()), ('model', Lasso())])
    glf = GridSearchCV(pipeline, param_grid={
        "model__alpha": np.arange(0.001, 0.01, 0.0001)}, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    glf.fit(X_train, y_train)
    with open("./output/glf.F.pkl", "wb") as f:
        pickle.dump(glf.best_estimator_, f)
    print("done")

    # モデルの評価
    print("----------------------------------------")
    pred_test = glf.best_estimator_.predict(X_test)
    print(
        f"CPに対するRMSE: {np.sqrt(mean_squared_error(y_test, pred_test))}")
    print(f"CPでの相関係数: {np.corrcoef(y_test, pred_test)[0, 1]}")

    df = pd.DataFrame({"pred": pred_test, "test": y_test})
    df["pred"] = df["pred"].apply(exercise_D.toPPB)
    df["test"] = df["test"].apply(exercise_D.toPPB)
    print(df)
    plt.scatter(df["pred"], df["test"])
    plt.xlabel(r"$CP\, Predicted\, f_b$")
    plt.ylabel(r"$CP\, f_b$")
    plt.savefig("./output/ex_F_test.png")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
