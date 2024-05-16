from data_preprocessing.remove_links import remove_links_and_empty_messages
from data_preprocessing.translation import translate_messages
from data_preprocessing.grouping import merge_consecutive_messages
import numpy as np
import pandas as pd

timeframe = 300


def select_data_for_labeling_conversation_id(df):
    df = remove_links_and_empty_messages(df)
    df = merge_consecutive_messages(df, timeframe=timeframe)
    # select random as there are lots of message
    new_df = pd.DataFrame()
    window_size = 300
    for group_link in df["group"].unique():
        group_df = df[df["group"] == group_link]
        group_df = group_df.sort_values(by=["date"])
        group_df = group_df.reset_index(drop=True)
        start_idx = np.random.randint(0, len(group_df) - window_size + 1)
        group_df = group_df.iloc[start_idx : start_idx + window_size]
        group_df = group_df.drop("Unnamed: 0", axis=1)
        new_df = pd.concat([new_df, group_df])
    new_df = new_df.reset_index(drop=True)
    df = translate_messages(new_df)
    return df


def select_data_for_labeling_topic(rf_model, df):
    from conversation_models.random_forest.data_preparation import (
        generate_dataset_from_labeled_data_with_sliding_window,
    )

    features_df = generate_dataset_from_labeled_data_with_sliding_window(
        df, window_size=4, message_id_column=True, label_column=False
    )
    messages_ids_df = features_df[["message_id"]]
    features_df = features_df.drop("message_id", axis=1)
    y = rf_model.predict(features_df)
    messages_ids_df["label"] = y
    roots_df = messages_ids_df[messages_ids_df["label"] == False]["message_id"]

    result_df = df[df["id"].isin(roots_df)]
    return result_df
