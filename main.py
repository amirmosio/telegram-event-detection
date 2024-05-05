import torch
from data_collection.fetching_data import fetch_messages_from_client_add_to_the_datafram
import pandas as pd
import asyncio
from decouple import config
from data_labeling.select_for_labeling import select_data_for_labeling
from decouple import config
from data_preprocessing.translation import translate_messages
from user_telegram_client.user_client import UserClient
import asyncio

from conversation_models.random_forest.data_preparation import (
    generate_dataset_from_labeled_data_with_sliding_window,
    split_train_test_validation_and_remove_extra_data,
)

from conversation_models.random_forest.model_training import (
    print_model_evaluation as print_model_evaluation_rf,
    train_random_forest_model,
    print_evaluation_on_test,
)
from conversation_models.neural_network.data_preparation import (
    generate_dataset_considering_root_of_conversation,
    replace_text_with_embedding,
    split_train_test_validation,
)
from conversation_models.neural_network.model_training import (
    train_neural_network_model,
    print_model_evaluation as print_model_evaluation_nn,
    draw_loss_chart,
)

commands = {
    "1": "Download dataset from telegram groups",
    "1.5": "To translate messages in trial.csv",
    "2": "Select labeling data",
    "3": "Train conversation model with random forest",
    "4": "Prepare training data for nn",
    "4.5": "Train conversation model with nn and embedding",
    "5": "Run Telegram client",
}
command = input(
    "Which phase:\n" + "\n".join([f"{key}-{commands[key]}" for key in commands]) + "\n"
)
commands[command]


if command == "1":
    batch_size = int(config("BATCH_SIZE"))
    batch_size_increase_rate = int(config("BATCH_SIZE_INCREASE_RATE"))
    time_interval_to_fetch_messages = int(config("TIME_INTERVAL_TO_FETCH_MESSAGES"))

    df = pd.DataFrame()

    groups_to_fetch_messages = [
        "https://t.me/joinchat/OOC4qk2QS1FM37aETHuzWQ",
        "https://t.me/joinchat/RL4pXSkXipyuKDmd",
        "https://t.me/joinchat/FNGD_0n6IpIbjfJBAZsuoA",
        "https://t.me/joinchat/qyxbq_vZ5f4xYzg0",
        "https://t.me/joinchat/rLRXuuItcHtkMTVk",
        "https://t.me/joinchat/aiAC6RgOjBRkYjhk",
        "https://t.me/PoliGruppo",
    ]
    for chat in groups_to_fetch_messages:
        print(f"fetching from {chat}")
        group_df = asyncio.run(
            fetch_messages_from_client_add_to_the_datafram(
                chat,
                limit=batch_size,
                batch_size_rate=batch_size_increase_rate,
                time_interval_to_fetch_messages=time_interval_to_fetch_messages,
            )
        )
        df = pd.concat([df, group_df])

    df.to_csv("./data/trial.csv")
elif command == "1.5":
    # to translate downloaded data
    raw_df = pd.read_csv("./data/trial.csv")
    raw_df = translate_messages(raw_df)
    raw_df.to_csv("./data/trial_translated.csv")

elif command == "2":
    raw_data = pd.read_csv("./data/trial.csv")
    df = select_data_for_labeling(raw_data)
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
    X_t, y_t, X_v, y_v, X_tv, y_tv = split_train_test_validation_and_remove_extra_data(
        dataset_df
    )
    # X_t.to_csv("./data/X_train.csv")
    # y_t.to_csv("./data/y_train.csv")

    # X_v.to_csv("./data/X_validation.csv")
    # y_v.to_csv("./data/y_validation.csv")

    # X_tv.to_csv("./data/X_test.csv")
    # y_tv.to_csv("./data/y_test.csv")

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
    print_evaluation_on_test(model, X_tv, y_tv)
elif command == "4":
    raw_data = pd.read_csv("./data/trial.csv")
    # raw_data = raw_data.iloc[:355]
    raw_data = generate_dataset_considering_root_of_conversation(
        raw_data, min_conversation_leng=4
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

elif command == "5":

    phone_number = config("TELEGRAM_CLIENT_PHONE_NUMBER")
    api_id = config("TELEGRAM_CLIENT_API_ID")
    api_hash = config("TELEGRAM_CLIENT_API_HASH")

    async def connect():
        client = UserClient(phone_number, api_id, api_hash)
        await client.connect(run_blocking=True)

    asyncio.run(connect())
