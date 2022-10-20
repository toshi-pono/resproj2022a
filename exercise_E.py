import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import loadData

print("loading...")
train_df, ans_df = loadData.load_data_from_smile_csv(
    "data/SM.csv", use_cache=True)


print("fitting...")
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', Lasso())])
glf = GridSearchCV(pipeline, param_grid={
                   "model__alpha": np.arange(0.001, 0.01, 0.0001)}, cv=5, scoring="neg_root_mean_squared_error")
glf.fit(train_df, ans_df)

print("done")

print(-glf.best_score_)
print(glf.best_params_)

result_df = pd.DataFrame(glf.cv_results_)
result_df.sort_values(
    by="rank_test_score", inplace=True)
print(result_df[["rank_test_score",
                 "params",
                 "mean_test_score"]])

# calc
best = glf.best_estimator_
y_pred = best.predict(train_df)
print(np.sqrt(mean_squared_error(ans_df, y_pred)))

# # 最良パラメータ
# print(f"Best Alpha (from MSE path):{clf.alpha_}")

# # print(clf.coef_)
# # print(clf.intercept_)

# # 最良パラメータにおける InKa のRMSE
# # TODO:
# print(
#     f"MSE at Best Alpha:{clf.mse_path_[np.argmin(clf.mse_path_.mean(axis=-1),)]}")
# # print(clf.mse_path_)
# y_pred = clf.predict(scaler.transform(train_df))
# mse = mean_squared_error(ans_df, y_pred)

# print(f"{mse=}")
