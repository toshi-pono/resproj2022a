import pickle
import pandas as pd
import exercise_C


def main():
    """
    構築したモデルから重要度を取得し、重要度の高い順に特徴量を並べる
    exercise_I.pyのmain関数を実行してから実行すること
    """
    with open("output/glf.pkl", "rb") as f:
        glf = pickle.load(f)
        importance = pd.DataFrame(
            glf.best_estimator_.named_steps["model"].feature_importances_, index=exercise_C.descriptor_names(), columns=["importance"])
        print(importance.sort_values(by="importance", ascending=False))


if __name__ == "__main__":
    main()
