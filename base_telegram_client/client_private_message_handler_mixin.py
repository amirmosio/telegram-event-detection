import telethon
from user_telegram_client.constants import Commands, Messages, Patterns
from user_telegram_client.models import User
import re


class ClientPrivateMessageHandlerMixin:
    def __init__(self) -> None:
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
                    and event.raw_text == Commands.Start
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
                elif event.raw_text == Commands.Groups:
                    await self.__handle_request_for_updating_groups(event)
                elif state == User.State.WAIT_GROUP_SET and all(
                    [
                        re.match(Patterns.Link, l)
                        for l in event.raw_text.strip().splitlines()
                    ]
                ):
                    await self.__handle_updating_groups(event)
                elif event.raw_text == Commands.Topics:
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
        groups = event.raw_text.strip().splitlines()
        groups = [l.strip("/") for l in groups]
        group_ids, invalid_link = await self.__join_groups(groups)
        if invalid_link:
            await event.reply(f"Invalid Link \n{invalid_link}")
        else:
            self.query_manager.update_group_for_user(
                telegram_id=event.sender_id, groups=group_ids
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
        groups = event.raw_text.strip().splitlines()
        groups = [l.strip("/") for l in groups]
        group_ids, invalid_link = await self.__join_groups(groups)
        if invalid_link:
            await event.reply(f"Invalid Link \n{invalid_link}")
        else:
            self.query_manager.update_group_for_user(
                telegram_id=event.sender_id, groups=group_ids
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

    async def __join_groups(self, groups):
        from telethon.tl.functions.messages import (
            ImportChatInviteRequest,
            CheckChatInviteRequest,
        )

        group_ids = []
        invalid_link = None
        for l in groups:
            hash_g = re.split("/|-|\+", l.strip("/"))[-1]
            try:
                updates = await self.client(ImportChatInviteRequest(hash_g))
                group_id = updates.chats[0].id
                group_ids.append(group_id)
            except telethon.errors.rpcerrorlist.UserAlreadyParticipantError:
                checked = await self.client(CheckChatInviteRequest(hash_g))
                group_id = checked.chat.id
                group_ids.append(group_id)
            except Exception as e:
                invalid_link = l
                return None, invalid_link
        return group_ids, None
