import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

import exercise_C

# TODO: getadta
print("loading...")
sm_df = pd.read_csv('data/SM.csv')
print("smiles to descriptor...")
descriptors_df = sm_df["SMILES"].map(exercise_C.SmileToDescriptorVec)

train_df = pd.DataFrame([row for row in descriptors_df])
ans_df = sm_df["PPB (fb)"]

print("fitting...")
scaler = StandardScaler()
clf = LassoCV(alphas=10 ** np.arange(-6, 1, 0.1), cv=5)

scaler.fit(train_df)
clf.fit(scaler.transform(train_df), ans_df)
# LassoCV(alphas=array([1.00000e-06, 1.2
# 5893e-06, ..., 6.30957e+00, 7.94328e+00]),
# copy_X=True, cv=5, eps=0.001, fit_intercept=True, max_iter=1000,
# n_alphas=100, n_jobs=None, normalize=False, positive=False,
# precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
# verbose=False)
print("done")

print(clf.alpha_)

print(clf.coef_)

print(clf.intercept_)
