from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_preprocessing.translation import translate_messages

from utilities.embeddings import (
    embedding_with_reoberta,
    embedding_with_sentence_transformer,
)


def generate_dataset_considering_root_of_conversation(df, min_conversation_leng=3):
    # minimume conversation length coresponds to number of nodes in tree of conversation
    # Here we assume that all the messages after start of conversation are replied to the root
    df = df.fillna(0).astype(
        {"reply": int, "id": int, "sender": int, "text": str}, errors="ignore"
    )
    df = df.sort_values(by="date", ascending=False)
    df["conversation_length_in_subtree"] = 0

    for idx in tqdm(range(len(df))):
        record = df.iloc[idx]
        if record["reply"]:
            parent_message_idx = df[df["id"] == record["reply"]].index
            df.loc[parent_message_idx, "conversation_length_in_subtree"] += (
                record["conversation_length_in_subtree"] + 1
            )
    df["is_root"] = False

    root_indcies = df[df["conversation_length_in_subtree"] >= min_conversation_leng]
    root_indcies = root_indcies[df["reply"] == 0].index

    df.loc[root_indcies, "is_root"] = True

    # Making data set balanced with respect to target value
    df = shuffle(df)
    false_df_to_be_dropped = df[df["is_root"] == False].iloc[
        sum(df["is_root"] == True) :
    ]
    df = df.drop(false_df_to_be_dropped.index)

    df = translate_messages(df)
    return df


def replace_text_with_embedding(df):
    embeddings = embedding_with_reoberta(df["text"].array)
    df["embedding"] = embeddings
    return df


def split_train_test_validation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
