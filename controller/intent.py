import pandas as pd

from feature.analysis_utils import plot_dataframe_subplots, get_pearson_matrix


def find_user_intent_dispose(intent_id, df_data: pd.DataFrame):
    if intent_id >= 3:
        raise RuntimeError("未找到对应执行的方法")
    if intent_id == 0:
        plot_dataframe_subplots(df_data, nrows=round(len(df_data.columns) + 1 / 4), ncols=4,
                                figsize=(30, len(df_data.columns) + 1))
    if intent_id == 1:
        get_pearson_matrix(df_data, figsize=(30, len(df_data.columns) + 1))
    if intent_id == 2:
        return


if __name__ == '__main__':
    df_master = pd.read_csv("../data/df_master.csv")
    find_user_intent_dispose(0, df_master)
