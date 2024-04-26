from data_preprocessing.remove_links import remove_links_and_empty_messages
from data_preprocessing.translation import translate_messages
from data_preprocessing.grouping import merge_consecutive_messages
import numpy as np
import pandas as pd

timeframe = 300

def select_data_for_labeling(df):
    df = remove_links_and_empty_messages(df)
    df = merge_consecutive_messages(df, timeframe=timeframe)
    # select random as there are lots of message
    new_df = pd.DataFrame()
    window_size = 300
    for group_link in df['group'].unique():
        group_df = df[df['group'] == group_link]
        group_df = group_df.sort_values(by=['date'])
        group_df = group_df.reset_index(drop=True)
        start_idx = np.random.randint(0, len(group_df) - window_size + 1)
        group_df = group_df.iloc[start_idx : start_idx + window_size]
        group_df = group_df.drop("Unnamed: 0", axis=1)
        new_df = pd.concat([new_df, group_df])
    new_df = new_df.reset_index(drop=True)
    df = translate_messages(new_df)
    return df