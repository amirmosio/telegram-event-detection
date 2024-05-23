from .data_collector_client import DataCollectorClient
from decouple import config
import pandas as pd
import datetime
from tqdm import tqdm

phone_number = config("TELEGRAM_CLIENT_PHONE_NUMBER")
api_id = config("TELEGRAM_CLIENT_API_ID")
api_hash = config("TELEGRAM_CLIENT_API_HASH")


async def fetch_messages_from_client_add_to_the_datafram(
    chat, limit, batch_size_rate=None, time_interval_to_fetch_messages=180
):

    client = DataCollectorClient(phone_number, api_id, api_hash)
    await client.connect(run_blocking=False)

    group_df = pd.DataFrame()

    last_message_date = datetime.datetime.now().date()
    last_message_date_is_in_desired_interval = False

    progress_days_bar = tqdm(range(time_interval_to_fetch_messages + 1))
    progress_days_bar.update(0)
    progress_days_bar.refresh()
    while not last_message_date_is_in_desired_interval:
        progress_days_bar.update(
            (datetime.datetime.now().date() - last_message_date).days
        )
        progress_days_bar.refresh()

        new_group_df = await client.get_messages(
            chat, limit, offset_date=last_message_date
        )
        if new_group_df.empty:
            break
        new_group_df = new_group_df.sort_values(by="date", ascending=True)
        last_message_date = pd.to_datetime(new_group_df.head(1)["date"]).dt.date.iloc[0]
        last_message_date_is_in_desired_interval = (
            last_message_date
            < (
                datetime.datetime.now()
                - datetime.timedelta(days=time_interval_to_fetch_messages)
            ).date()
        )

        group_df = pd.concat([group_df, new_group_df])

        limit *= batch_size_rate
    await client.disconnect()
    return group_df
