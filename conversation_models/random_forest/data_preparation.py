import json
import pandas as pd
import torch
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
)
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np


def calculate_time_delta(a, b):
    datetime_format = "%Y-%m-%d %H:%M:%S%z"
    a_datetime = datetime.strptime(a, datetime_format)
    b_datetime = datetime.strptime(b, datetime_format)
    time_delta = a_datetime - b_datetime
    return time_delta / timedelta(minutes=1)


def emobedding_with_sentence_transformer(messages):
    emodel = SentenceTransformer("average_word_embeddings_glove.6B.300d")  # 300
    return list(emodel.encode(messages))


# def embedding_with_laser(messages):
#     from laserembeddings import Laser

#     laser = Laser()
#     return laser.embed_sentences(messages, lang=["en"] * len(len(messages)))


def embedding_with_reoberta(messages):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def calculate_embedding(m):
        input_tokens = tokenizer(
            m,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        return model(**input_tokens)

    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(messages))):
            try:
                output = calculate_embedding(messages[i])
            except:
                output = calculate_embedding("")

            embeddings.append(output.last_hidden_state[:, 0, :])
    return embeddings


def generate_dataset_from_labeled_data_with_sliding_window(df, window_size=5):
    embeddings = emobedding_with_sentence_transformer(df["text"])
    df["embedding"] = embeddings
    result_df = pd.DataFrame()
    for start_idx in tqdm(range(len(df) - window_size - 1)):
        record_df = df.iloc[start_idx : start_idx + window_size + 1]
        last_messages_conversation = record_df.iloc[-1]["conversation_id"]

        window_df = record_df.iloc[:-1]

        train_record = {
            f"reaction_ref": sum(
                json.loads(record_df.iloc[-1]["reactions"].replace("'", '"')).values()
            )
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
    return result_df


def split_train_test_validation_and_remove_extra_data(df):
    # To make the dataset more balanced and unbiased we have to drop extra label==True records so that
    # number of label==True traning would be equal to number of label=False
    df = df.drop("index", axis=1)
    true_df_to_be_dropped = df[df["label"] == True].iloc[sum(df["label"] == False) :]
    new_df = df.drop(true_df_to_be_dropped.index)
    X = new_df.drop("label", axis=1)
    y = new_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test