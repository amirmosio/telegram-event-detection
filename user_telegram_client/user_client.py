from imaplib import Commands
import telethon
from base_telegram_client.client import TelegramClient
from decouple import config
import asyncio
import re
from .models import User
from .manager import QueryManager
from user_telegram_client.Constants import Messages, Patterns
from telethon.tl.types import KeyboardButtonUrl
from telethon.tl.custom import Button


class UserClient(TelegramClient):
    class Commands:
        Start = "start"
        Groups = "group"
        Topics = "topic"

    def __init__(self, phone_number, api_id, api_hash):
        super(UserClient, self).__init__(phone_number, api_id, api_hash)
        self.query_manager = QueryManager()

        @self.client.on(
            telethon.events.NewMessage(
                func=lambda e: not e.is_group and not e.is_channel and e.is_private
            )
        )
        async def private_message_handler(event):
            a = event.from_id, event.peer_id, event.sender_id
            sender = await event.get_sender()
            sender.phone, sender.id
            if self.query_manager.check_if_user_has_been_here_before(event.sender_id):
                state = self.query_manager.get_user_state(event.sender_id)
                if (
                    state == User.State.NOT_REGISTERED
                    and event.raw_text == self.Commands.Start
                ):
                    await self.__handle_not_registered_user(event)
                elif state == User.State.REGISTERED and all(
                    [
                        re.match(Patterns.Link, l)
                        for l in event.raw_text.strip().splitlines()
                    ]
                ):
                    await self.__handle_setting_groups(event)
                elif state == User.State.GROUPS_SET and all(
                    [
                        re.match(Patterns.TopicIds, l)
                        for l in event.raw_text.strip().splitlines()
                    ]
                ):
                    await self.__handle_setting_topics(event)
                elif event.raw_text == self.Commands.Groups:
                    await self.__handle_request_for_updating_groups(event)
                elif state == User.State.WAIT_GROUP_SET and all(
                    [
                        re.match(Patterns.Link, l)
                        for l in event.raw_text.strip().splitlines()
                    ]
                ):
                    await self.__handle_updating_groups(event)
                elif event.raw_text == self.Commands.Topics:
                    await self.__handle_requiest_for_updating_topics(event)
                elif state == User.State.WAIT_TOPIC_SET and all(
                    [
                        re.match(Patterns.TopicIds, l)
                        for l in event.raw_text.strip().splitlines()
                    ]
                ):
                    await self.__handle_updating_topics(event)
                else:
                    await event.reply(Messages.INVALID_INPUT)

            else:
                self.query_manager.create_user(
                    telegram_id=event.sender_id,
                    username=sender.username,
                    first_name=sender.first_name,
                    last_name=sender.last_name,
                )
                await event.reply(Messages.WELCOMING)

    async def __handle_not_registered_user(self, event):
        await event.reply(Messages.SEND_GROUPS_LINK)
        self.query_manager.update_user_state(event.sender_id, User.State.REGISTERED)

    async def __handle_setting_groups(self, event):
        self.query_manager.update_group_for_user(
            telegram_id=event.sender_id, groups=event.raw_text.strip().splitlines()
        )
        self.query_manager.update_user_state(event.sender_id, User.State.GROUPS_SET)
        await event.reply(Messages.SEND_INTERESTED_TOPICS)

    async def __handle_setting_topics(self, event):
        self.query_manager.update_topics_for_user(
            telegram_id=event.sender_id, topics=event.raw_text.strip().splitlines()
        )
        self.query_manager.update_user_state(event.sender_id, User.State.TOPIC_SET)
        await event.reply(Messages.FINAL)

    async def __handle_request_for_updating_groups(self, event):
        self.query_manager.update_user_state(event.sender_id, User.State.WAIT_GROUP_SET)
        await event.reply(Messages.SEND_GROUPS_LINK)

    async def __handle_updating_groups(self, event):
        self.query_manager.update_group_for_user(
            telegram_id=event.sender_id, groups=event.raw_text.strip().splitlines()
        )
        self.query_manager.update_user_state(event.sender_id, User.State.TOPIC_SET)
        await event.reply(Messages.DONE)

    async def __handle_requiest_for_updating_topics(self, event):
        self.query_manager.update_user_state(event.sender_id, User.State.WAIT_TOPIC_SET)
        await event.reply(Messages.SEND_INTERESTED_TOPICS)

    async def __handle_updating_topics(self, event):
        self.query_manager.update_topics_for_user(
            telegram_id=event.sender_id, topics=event.raw_text.strip().splitlines()
        )
        self.query_manager.update_user_state(event.sender_id, User.State.TOPIC_SET)
        await event.reply(Messages.DONE)

    async def get_messages(self, chat_id, limit=None, offset_date=None):
        async with self.client.takeout() as takeout:
            return await takeout.get_messages(
                chat_id, limit=limit, offset_date=offset_date
            )  # wrapped through takeout (less limits)
