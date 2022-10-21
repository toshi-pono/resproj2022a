import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import loadData
import pickle

def main():
    print("loading...")
    train_df, ans_df = loadData.load_data_from_smile_csv(
        "data/SM.csv", use_cache=True)
    test_df, test_ans_df = loadData.load_data_from_smile_csv(
        "data/CP.csv", use_cache=True)

    print("fitting...")
    param_grid = {'model__max_depth': np.arange(2, 15, 1),
              'model__min_samples_leaf': np.arange(1, 10, 1)}
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', RandomForestRegressor())])
    glf = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    glf.fit(train_df, ans_df)

    # save glf
    with open("output/glf.pkl", "wb") as f:
        pickle.dump(glf, f)

    print("done")

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
    test_pred = glf.best_estimator_.predict(test_df)
    print(f"テストデータ全体に対するRMSE: {np.sqrt(mean_squared_error(test_ans_df, test_pred))}")
    print(f"相関係数: {np.corrcoef(test_ans_df, test_pred)[0, 1]}")
    print("----------------------------------------")

if __name__ == "__main__":
    main()