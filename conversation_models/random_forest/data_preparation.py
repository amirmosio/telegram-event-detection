import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from conversation_models.neural_network.model_training import ConversationRootModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
from utilities.embeddings import (
    embedding_with_reoberta,
    embedding_with_sentence_transformer,
)


def calculate_time_delta(a, b):
    datetime_format = "%Y-%m-%d %H:%M:%S%z"
    a_datetime = datetime.strptime(a, datetime_format)
    b_datetime = datetime.strptime(b, datetime_format)
    time_delta = a_datetime - b_datetime
    return time_delta / timedelta(minutes=1)


@torch.no_grad()
def generate_dataset_from_labeled_data_with_sliding_window(df, window_size=5):
    embeddings = embedding_with_reoberta(df["text"])
    df["embedding"] = embeddings
    result_df = pd.DataFrame()

    for start_idx in tqdm(range(len(df) - window_size - 1)):
        record_df = df.iloc[start_idx : start_idx + window_size + 1]
        last_messages_conversation = record_df.iloc[-1]["conversation_id"]

        window_df = record_df.iloc[:-1]

        train_record = {
            f"reaction_ref": sum(
                json.loads(record_df.iloc[-1]["reactions"].replace("'", '"')).values()
            ),
            "nn_embedding": record_df.iloc[-1]["embedding"],
        }
        for i in range(window_size):
            train_record |= {
                f"reaction{i}": sum(
                    json.loads(
                        window_df.iloc[i]["reactions"].replace("'", '"')
                    ).values()
                )
            }
            train_record |= {
                f"d{i}": np.linalg.norm(
                    window_df.iloc[i]["embedding"] - record_df.iloc[-1]["embedding"]
                )
            }
            train_record |= {
                f"timegap{i}": calculate_time_delta(
                    record_df.iloc[-1]["date"], window_df.iloc[i]["date"]
                )
            }
            same_profile = False
            try:
                same_profile = int(record_df.iloc[-1]["sender"]) == int(
                    window_df.iloc[i]["sender"]
                )
            except:
                pass
            train_record |= {f"same_profile{i}": same_profile}

        # Setting Target Values
        train_record |= {
            "label": last_messages_conversation in window_df["conversation_id"].array
        }
        result_df = result_df._append(train_record, ignore_index=True)

    result_df = result_df.reset_index()

    model = ConversationRootModel(input_feature_size=len(embeddings[-1]))
    model.load_model("conversation_model_1715021680.755679_epoch_24_acc_83")

    nn_embeddings = np.array([x for x in result_df["nn_embedding"]])
    nn_embeddings = torch.tensor(nn_embeddings, dtype=torch.float)
    result_df["nn_embedding"] = model(nn_embeddings)

    result_df = result_df.drop("index", axis=1)

    # To make the dataset more balanced and unbiased we have to drop extra label==True records so that
    # number of label==True traning would be equal to number of label=False

    # true_df_to_be_dropped = result_df[result_df["label"] == True].iloc[
    #     sum(result_df["label"] == False) :
    # ]
    # result_df = result_df.drop(true_df_to_be_dropped.index)
    return result_df
