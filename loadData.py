import pandas as pd
import exercise_C
import exercise_D


def load_data_from_smile_csv(file: str, use_cache=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a csv file with the following columns:
    SMILES,PPB(fb)

    Parameters
    ----------
    file : str
        The path to the csv file
    use_cache : bool, optional
        If True, use cache, by default False

    Returns
    -------
    data: tuple[pd.DataFrame, pd.DataFrame]
        The first element is the input data, the second element is the answer data
    """
    if use_cache:
        try:
            input_df = pd.read_pickle(file + ".input.pkl")
            answer_df = pd.read_pickle(file + ".ans.pkl")
            return (input_df, answer_df)
        except FileNotFoundError:
            pass

    smile_df = pd.read_csv(file)
    descriptors_df = smile_df["SMILES"].map(exercise_C.smiles_to_descriptors)
    input_df = pd.DataFrame([row for row in descriptors_df])
    answer_df = smile_df["PPB(fb)"].map(parse_PPB).map(exercise_D.calc_lnka)

    # save cache
    input_df.to_pickle(file + ".input.pkl")
    answer_df.to_pickle(file + ".ans.pkl")
    return (input_df, answer_df)


def parse_PPB(fb: str | float) -> float:
    """
    Parse the PPB(fb) column
    """
    if isinstance(fb, float):
        return fb
    if isinstance(fb, str):
        if is_float(fb):
            return float(fb)
        else:
            s = fb.split("-")
            return (float(s[0]) + float(s[1])) / 2


def is_float(s: str) -> bool:
    """
    Check if a string is a float
    """
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
