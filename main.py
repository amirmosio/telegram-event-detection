import json
import torch
from data_collection.fetching_data import fetch_messages_from_client_add_to_the_datafram
import pandas as pd
import asyncio
from decouple import config
from data_labeling.select_for_labeling import (
    select_data_for_labeling_conversation_id,
    select_data_for_labeling_topic,
)
from sklearn.utils import shuffle
import joblib
from decouple import config
from data_preprocessing.translation import translate_messages
from user_telegram_client.user_client import UserClient
import asyncio
from sklearn.model_selection import train_test_split
from data_preprocessing.remove_links import remove_links_and_empty_messages

from conversation_models.random_forest.data_preparation import (
    generate_dataset_from_labeled_data_with_sliding_window,
)

from conversation_models.random_forest.model_training import (
    print_model_evaluation as print_model_evaluation_rf,
    train_random_forest_model,
    print_evaluation_on_test_plus_draw_sample_charts,
)
from conversation_models.neural_network.data_preparation import (
    generate_dataset_considering_root_of_conversation,
    replace_text_with_embedding,
)
from utilities.general import split_train_test_validation
from conversation_models.neural_network.model_training import (
    ConversationRootModel,
    train_neural_network_model,
    print_model_evaluation as print_model_evaluation_nn,
    draw_loss_chart,
)
from conversation_models.random_forest_categories.model_training import (
    train_random_forest_model_multiclass,
    print_model_evaluation,
)
from conversation_models.neural_network_categories.neural_network_categories import (
    neural_network,
)
from conversation_models.keyword_search_categories.keyword_search_categories import (
    get_categories_searching_through_keywrods,
    print_accuracy,
)
from utilities.finding_synonyms import get_synonyms

commands = {
    "1": "Download dataset from telegram groups",
    "1.5": "To translate messages in trial.csv",
    "2": "Select labeling data",
    "3": "Train conversation model with random forest",
    "4": "Prepare training data for nn",
    "4.5": "Train conversation model with nn and embedding",
    "4.6": "Test final nn model with test data",
    "4.8": "Select labeling data for topic and categories",
    "5.1": "Train and test random forest for topics",
    "5.2": "Train and test random nueral network for topics",
    "5.2.9": "Extract all keywords from topic labeled datasets",
    "5.2.9.5": "Add synonyms to all keywords in topic keywords mapping",
    "5.3": "The keyword method for topic modeling",
    "6": "Run Telegram client",
}
command = input(
    "Which phase:\n" + "\n".join([f"{key}-{commands[key]}" for key in commands]) + "\n"
)
commands[command]

DEBUG = batch_size = bool(config("DEBUG"))

if command == "1":
    batch_size = int(config("BATCH_SIZE"))
    batch_size_increase_rate = int(config("BATCH_SIZE_INCREASE_RATE"))
    time_interval_to_fetch_messages = int(config("TIME_INTERVAL_TO_FETCH_MESSAGES"))

    df = pd.DataFrame()

    groups_to_fetch_messages = [
        "https://t.me/joinchat/UD-gw7Ff7BgxOTQ0",
        "https://t.me/joinchat/yKfqfy4PH49lNDg0",
        "https://t.me/joinchat/8Z9Y0f3ymPw3NzNk",
        "https://t.me/joinchat/QeBrjx97pwI4NDVk",
        "https://t.me/joinchat/LoATRt0pfH81MzBk",
        "https://t.me/joinchat/zYSU3QDIPppkZmQ0",
        "https://t.me/joinchat/LGwn9L_TtK82OTVk",
        "https://t.me/joinchat/m8gLUqr5h2dlMTdk",
        "https://t.me/joinchat/LU4bgriZYWs5Yzg8",
        "https://t.me/joinchat/TXRLn8NjlbA0NzQ0",
        # "https://t.me/joinchat/OOC4qk2QS1FM37aETHuzWQ",
        # "https://t.me/joinchat/RL4pXSkXipyuKDmd",
        # "https://t.me/joinchat/FNGD_0n6IpIbjfJBAZsuoA",
        # "https://t.me/joinchat/qyxbq_vZ5f4xYzg0",
        # "https://t.me/joinchat/rLRXuuItcHtkMTVk",
        # "https://t.me/joinchat/aiAC6RgOjBRkYjhk",
        # "https://t.me/PoliGruppo",
    ]
    for chat in groups_to_fetch_messages:
        print(f"fetching from {chat}")
        group_df = asyncio.run(
            fetch_messages_from_client_add_to_the_datafram(
                chat,
                limit=batch_size,
                batch_size_rate=batch_size_increase_rate,
                time_interval_to_fetch_messages=100,
            )
        )
        df = pd.concat([df, group_df])

    df.to_csv("./data/trial.csv")
elif command == "1.5":
    # to translate downloaded data
    # raw_df = pd.read_csv("./data/trial.csv")
    # # raw_df = raw_df.iloc[:555]
    # raw_df = translate_messages(raw_df)
    # raw_df.to_csv("./data/trial_translated.csv")
    raw_df = pd.read_csv("./data/dataset_for_topic_labeling.csv")
    raw_df = raw_df.sample(frac=1).sort_values(by="topic")

    raw_df.to_csv("./data/dataset_for_topic_labeling.csv")
elif command == "2":
    raw_data = pd.read_csv("./data/trial.csv")
    df = select_data_for_labeling_conversation_id(raw_data)
    df.to_csv("./data/labeling_data.csv")

elif command == "3":
    raw_data = pd.read_csv(
        "./data/labeled_data.csv", sep=";", encoding="unicode_escape"
    )
    raw_data = raw_data.dropna(subset=["conversation_id"])
    raw_data = raw_data.reset_index(drop=True)
    # raw_data = raw_data.iloc[:55]
    dataset_df = generate_dataset_from_labeled_data_with_sliding_window(
        raw_data, window_size=4
    )
    X = dataset_df.drop("label", axis=1)
    y = dataset_df["label"]
    X_t, y_t, X_v, y_v, X_tv, y_tv = split_train_test_validation(
        X, y, test_ratio=0.2, val_ratio=0.2
    )
    X_t.to_csv("./data/X_train.csv")
    y_t.to_csv("./data/y_train.csv")

    X_v.to_csv("./data/X_validation.csv")
    y_v.to_csv("./data/y_validation.csv")

    X_tv.to_csv("./data/X_test.csv")
    y_tv.to_csv("./data/y_test.csv")

    model = train_random_forest_model(X_t, y_t)
    print("important features")
    importances = model.feature_importances_
    columns_enumeration = [(column, i) for i, column in enumerate(X_t.columns)]
    columns_enumeration.sort()
    for column, i in columns_enumeration:
        print(f"{column} {round(importances[i], ndigits=3)}", end=", ")
    print("On Validation:")
    print_model_evaluation_rf(model, X_v, y_v)
    print("On Test:")
    print_model_evaluation_rf(model, X_tv, y_tv)

    # joblib.dump(model, f"./trained_models/RF_93.joblib")
elif command == "4":
    raw_data = pd.read_csv("./data/trial.csv")
    # raw_data = raw_data.iloc[:355]
    raw_data = generate_dataset_considering_root_of_conversation(
        raw_data, min_conversation_leng=3
    )
    raw_data.to_csv("./data/training_data_for_nn.csv")

elif command == "4.5":
    torch.manual_seed(0)
    raw_data = pd.read_csv("./data/training_data_for_nn.csv")
    # raw_data = raw_data.iloc[:55]
    raw_data = replace_text_with_embedding(raw_data)
    X_t, y_t, X_v, y_v, X_tv, y_tv = split_train_test_validation(
        raw_data["embedding"], raw_data["is_root"]
    )

    model, train_acc_history, val_acc_history = train_neural_network_model(
        X_t, y_t, X_v, y_v
    )
    print_model_evaluation_nn(model, X_v, y_v)
    draw_loss_chart(train_acc_history, val_acc_history)
elif command == "4.6":
    torch.manual_seed(0)
    raw_data = pd.read_csv("./data/training_data_for_nn.csv")
    raw_data = raw_data.iloc[1879:1880]
    # _, _, _, _, X_tv, y_tv = split_train_test_validation(
    #     raw_data[["text"]], raw_data["is_root"]
    # )
    X_tv, y_tv = raw_data[["text"]], raw_data["is_root"]
    X_tv = replace_text_with_embedding(X_tv)["embedding"]

    with torch.no_grad():
        model = ConversationRootModel(input_feature_size=len(X_tv.iloc[-1]))
        model.load_model("conversation_model_1715021680.755679_epoch_24_acc_83")
        model.eval()
        print("Testing")
        print_model_evaluation_nn(model, X_tv, y_tv)
elif command == "4.8":
    # Select data for labeling topics
    raw_data = pd.read_csv("./data/trial_translated.csv")
    # raw_data = raw_data.iloc[8:333]
    # load model
    model = joblib.load("./trained_models/RF_93.joblib")
    result_df = select_data_for_labeling_topic(model, raw_data)
    result_df.to_csv("./data/dataset_for_topic_labeling.csv")

elif command == "5.1":

    raw_data = pd.read_csv("./data/dataset_for_topic_labeling.csv")
    raw_preprocessed = remove_links_and_empty_messages(raw_data)
    raw_preprocessed.dropna(
        subset=["topic"], inplace=True
    )  # dataset is labelled only partially
    X = raw_preprocessed["text"]
    y = raw_preprocessed["topic"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    clf_multiclass = train_random_forest_model_multiclass(X_train, y_train)
    print_model_evaluation(clf_multiclass, X_test, y_test)

elif command == "5.2":

    raw_data = pd.read_csv(
        "./data/NNTopic_dataframe.csv"
    )  # dataset is labelled only partially
    raw_preprocessed = remove_links_and_empty_messages(raw_data)
    raw_preprocessed.dropna(subset=["topic"], inplace=True)
    X = raw_data["text"]
    y = raw_data["topic"]
    neural_network(X, y)

elif command == "5.2.9":
    raw_data = pd.read_csv("./data/dataset_for_topic_labeling.csv")
    topic_keyword_df = raw_data.dropna(subset=["topic", "keywords"])
    topic_keyword_dict = {}
    for idx in range(len(topic_keyword_df)):
        r = topic_keyword_df.iloc[idx]
        topic = r["topic"]
        keywords = [k.strip() for k in r["keywords"].split(",")]
        if topic not in topic_keyword_dict:
            topic_keyword_dict[topic] = set()
        topic_keyword_dict[topic].update(keywords)
    print(topic_keyword_dict)
elif command == "5.2.9.5":
    with open("topic_keywords_dict.json") as f:
        topic_keyword_dict = json.load(f)
    for key in topic_keyword_dict:
        new_keywords = []
        for w in topic_keyword_dict[key]:
            new_keywords.append(w)
            new_keywords += get_synonyms(w)
        topic_keyword_dict[key] = list(set(new_keywords))
    with open("topic_keywords_dict.json", "w") as f:
        # Dump the dictionary into the file
        json.dump(topic_keyword_dict, f)

elif command == "5.3":
    raw_data = pd.read_csv("./data/dataset_for_topic_labeling.csv")
    raw_data = raw_data.dropna(subset=["topic", "text"])

    raw_data = raw_data[raw_data["reply"].isnull()]
    raw_data = shuffle(raw_data)

    # raw_data = raw_data.iloc[1:433]
    with open("topic_keywords_dict.json") as f:
        topic_keyword_dict = json.load(f)
    sample_texts = ["Is the score from 33?"]
    categories = get_categories_searching_through_keywrods(
        raw_data["text"], topic_keyword_dict
    )
    raw_data["predicted_topic"] = categories

    print_accuracy(raw_data["predicted_topic"], raw_data["topic"])
elif command == "6":

    phone_number = config("TELEGRAM_CLIENT_PHONE_NUMBER")
    api_id = config("TELEGRAM_CLIENT_API_ID")
    api_hash = config("TELEGRAM_CLIENT_API_HASH")

    async def connect():
        client = UserClient(phone_number, api_id, api_hash)
        await client.connect(run_blocking=True)

    asyncio.run(connect())
