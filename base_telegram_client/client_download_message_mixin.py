import pandas as pd
import json


class ClientDownloadMessageMixin:
    async def __get_messages_from_session(
        self, session, chat_id, limit=None, offset_date=None, offset_id=None
    ):
        messages = await session.get_messages(
            chat_id, limit=limit, offset_date=offset_date, offset_id=offset_id
        )  # wrapped through takeout (less limits)
        new_group_df = pd.DataFrame()
        for message in messages:
            reactions_result = (
                [] if message.reactions is None else message.reactions.results
            )
            # Add reactions as feature
            reactions_dict = {}
            for r in reactions_result:
                try:
                    reactions_dict[r.reaction.emoticon] = r.count
                except:
                    pass

            data = {
                "id": message.id,
                "group": chat_id,
                "sender": message.sender_id,
                "text": message.text,
                "reply": message.reply_to_msg_id,
                "date": message.date,
                "reactions": json.dumps(reactions_dict),
            }
            new_group_df = new_group_df._append(data, ignore_index=True)
        return new_group_df

    async def get_messages_without_takeout(
        self, chat_id, limit=None, offset_date=None, offset_id=None
    ):
        return await self.__get_messages_from_session(
            self.client,
            chat_id,
            limit=limit,
            offset_date=offset_date,
            offset_id=offset_id,
        )

    async def get_messages(self, chat_id, limit=None, offset_date=None, offset_id=None):
        async with self.client.takeout() as takeout:
            return await self.__get_messages_from_session(
                takeout,
                chat_id,
                limit=limit,
                offset_date=offset_date,
                offset_id=offset_id,
            )
