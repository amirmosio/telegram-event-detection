from data_preprocessing.remove_links import remove_links
from data_preprocessing.translation import translate_messages
import numpy as np
import pandas as pd

def select_data_for_labeling(df):
    df = remove_links(df)
    # select random as there are lots of message
    new_df = pd.DataFrame()
    window_size = 300
    for group_link in df['group'].unique():
        group_df = df.loc[df['group'] == group_link]
        start_idx = np.random.randint(0, len(group_df) - window_size + 1)
        group_df = group_df.iloc[start_idx : start_idx + window_size]
        group_df = group_df.drop("Unnamed: 0", axis=1)
        new_df = pd.concat([new_df, group_df])
    new_df = new_df.reset_index(drop=True)
    df = translate_messages(new_df)
    return df