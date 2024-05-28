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
    embedding_with_distil_bert,
    embedding_with_sentence_transformer,
)
from settings import DEBUG


def calculate_time_delta(a, b):
    if type(a) == str and type(b) == str:
        datetime_format = "%Y-%m-%d %H:%M:%S%z"
        a_datetime = datetime.strptime(a, datetime_format)
        b_datetime = datetime.strptime(b, datetime_format)
    else:
        a_datetime = a
        b_datetime = b
    time_delta = a_datetime - b_datetime
    return time_delta / timedelta(minutes=1)


@torch.no_grad()
def generate_dataset_from_labeled_data_with_sliding_window(
    raw_df,
    window_size=5,
    message_id_column=False,
    label_column=True,
    embedding_tokenizer=None,
    embedding_model=None,
    conversation_model_nn=None,
):
    df = raw_df.copy(deep=True)
    embeddings = embedding_with_distil_bert(
        df["text"], tokenizer=embedding_tokenizer, model=embedding_model
    )
    df["embedding"] = embeddings
    result_df = pd.DataFrame()

    result_df["greetings"] = 0
    result_df["thanks"] = 0
    result_df["questions"] = 0

    greet = [
        "hi",
        "hi!",
        "hi,",
        "hi.",
        "hello",
        "hello!",
        "hello,",
        "hello.",
        "hey",
        "hey!",
        "hey,",
        "hey.",
    ]
    greet2 = [
        "good morning",
        "good morning!",
        "good morning,",
        "good morning.",
        "good afternoon",
        "good afternoon!",
        "good afternoon,",
        "good afternoon.",
        "good evening",
        "good evening!",
        "good evening,",
        "good evening.",
    ]
    thank = ["thanks", "thanks!", ".thanks", "thank", ".thank"]

    for start_idx in tqdm(range(len(df) - window_size)):
        record_df = df.iloc[start_idx : start_idx + window_size + 1]
        if label_column:
            last_messages_conversation = record_df.iloc[-1]["conversation_id"]

        window_df = record_df.iloc[:-1]

        train_record = {
            f"reaction_ref": sum(
                json.loads(
                    record_df.iloc[-1]["reactions"]
                    .replace("'", '"')
                    .encode("utf-8")
                    .decode("unicode_escape")
                ).values()
            ),
            "nn_embedding": record_df.iloc[-1]["embedding"],
            "thanks": 0.0,
            "questions": 0.0,
            "greetings": 0.0,
        }
        if message_id_column:
            train_record["message_id"] = record_df.iloc[-1]["id"]
        text = str(record_df.iloc[-1]["text"]).lower()
        words = text.split()
        text_with_spaces = " ".join(words)

        if any(word in greet for word in words) and pd.isnull(
            record_df.iloc[-1]["reply"]
        ):
            train_record["greetings"] = 1.0
        if any(phrase in text_with_spaces for phrase in greet2) and pd.isnull(
            record_df.iloc[-1]["reply"]
        ):
            train_record["greetings"] = 1.0

        # check if it's a thanks message without a '?' & 'thanks in advance'
        if (
            any(word in thank for word in words)
            and not any("?" in word for word in words)
            and not any("advance" in word for word in words)
        ):
            train_record["thanks"] = 1.0

        # check if it's a question
        if "?" in text:
            train_record["questions"] = 1.0

        for i in range(window_size):
            train_record |= {
                f"reaction{i}": sum(
                    json.loads(
                        window_df.iloc[i]["reactions"]
                        .replace("'", '"')
                        .replace("'", '"')
                        .encode("utf-8")
                        .decode("unicode_escape")
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
                + (15 if DEBUG else 0)  # TODO Just for testing
            }
            same_profile = False
            try:
                same_profile = int(record_df.iloc[-1]["sender"]) == int(
                    window_df.iloc[i]["sender"]
                )
            except:
                pass
            train_record |= {f"same_profile{i}": same_profile}

        if label_column:
            # Setting Target Values
            train_record |= {
                "label": last_messages_conversation
                in window_df["conversation_id"].array
            }
        result_df = result_df._append(train_record, ignore_index=True)

    result_df = result_df.reset_index()

    if conversation_model_nn is None:
        conversation_model_nn = ConversationRootModel(
            input_feature_size=len(embeddings[-1])
        )
        conversation_model_nn.load_model(
            "conversation_model_1715021680.755679_epoch_24_acc_83"
        )
        conversation_model_nn.eval()
    nn_embeddings = np.array([x for x in result_df["nn_embedding"]])
    nn_embeddings = torch.tensor(nn_embeddings, dtype=torch.float)

    result_df["nn_embedding"] = conversation_model_nn(nn_embeddings)

    result_df = result_df.drop("index", axis=1)

    # To make the dataset more balanced and unbiased we have to drop extra label==True records so that
    # number of label==True traning would be equal to number of label=False

    # true_df_to_be_dropped = result_df[result_df["label"] == True].iloc[
    #     sum(result_df["label"] == False) :
    # ]
    # result_df = result_df.drop(true_df_to_be_dropped.index)
    return result_df
