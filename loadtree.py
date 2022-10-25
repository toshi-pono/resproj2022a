import pickle
import numpy as np
import pandas as pd

with open("output/glf.pkl", "rb") as f:
    glf = pickle.load(f)
    print(glf.best_estimator_.named_steps["model"].feature_importances_)