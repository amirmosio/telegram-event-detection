from .data_collector_client import DataCollectorClient
from decouple import config
import pandas as pd
import datetime

phone_number = config('TELEGRAM_CLIENT_PHONE_NUMBER')
api_id = config('TELEGRAM_CLIENT_API_ID')
api_hash = config('TELEGRAM_CLIENT_API_HASH')

async def fetch_messages_from_client_add_to_the_datafram(chat, limit, batch_size_rate=None, time_interval_to_fetch_messages=180):

    client = DataCollectorClient(phone_number, api_id, api_hash)
    await client.connect(run_blocking=False)

    group_df = pd.DataFrame()

    last_message_date = datetime.datetime.now()
    last_message_date_is_in_desired_interval = False

    while not last_message_date_is_in_desired_interval:

        messages = await client.get_messages(chat, limit, offset_date=last_message_date)
        new_group_df = pd.DataFrame()
        for message in messages:
            reactions_result = [] if message.reactions is None else message.reactions.results
            reactions_dict = {r.reaction.emoticon:r.count for r in reactions_result}
            data = { "group" : chat, "sender" : message.sender_id, "text" : message.text, "reply" : message.reply_to_msg_id, "date" : message.date, "reactions" : reactions_dict}
            new_group_df = new_group_df._append(data, ignore_index = True) 

        if new_group_df.empty:
            break
        new_group_df = new_group_df.sort_values(by="date", ascending=True)
        last_message_date = pd.to_datetime(new_group_df.head(1)['date']).dt.date.iloc[0]
        last_message_date_is_in_desired_interval = last_message_date < (datetime.datetime.now()-datetime.timedelta(days=time_interval_to_fetch_messages)).date()
        

        group_df = pd.concat([group_df, new_group_df])

        limit *= batch_size_rate
    
    await client.disconnect()
    return group_df
    
