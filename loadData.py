import pandas as pd
import exercise_C
import exercise_D


def load_data_from_smile_csv(file: str, use_cache=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a csv file with the following columns:
    SMILES,PPB (fb)
    """
    if use_cache:
        try:
            train_df = pd.read_pickle(file + ".train.pkl")
            ans_df = pd.read_pickle(file + ".ans.pkl")
            return (train_df, ans_df)
        except FileNotFoundError:
            pass

    smile_df = pd.read_csv(file)
    descriptors_df = smile_df["SMILES"].map(exercise_C.smile_to_descriptor_vec)
    train_df = pd.DataFrame([row for row in descriptors_df])
    ans_df = smile_df["PPB (fb)"].map(exercise_D.calc_lnka)

    # save cache
    train_df.to_pickle(file + ".train.pkl")
    ans_df.to_pickle(file + ".ans.pkl")
    return (train_df, ans_df)
