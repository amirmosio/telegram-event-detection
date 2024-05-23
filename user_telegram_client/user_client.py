from base_telegram_client.client import TelegramClient
from base_telegram_client.client_group_message_handler_mixin import (
    ClientGroupMessageHandlerMixin,
)
from base_telegram_client.client_private_message_handler_mixin import (
    ClientPrivateMessageHandlerMixin,
)
from base_telegram_client.client_download_message_mixin import (
    ClientDownloadMessageMixin,
)
from .manager import QueryManager


class UserClient(
    TelegramClient,
    ClientPrivateMessageHandlerMixin,
    ClientGroupMessageHandlerMixin,
    ClientDownloadMessageMixin,
):
    def __init__(self, phone_number, api_id, api_hash):
        super(UserClient, self).__init__(phone_number, api_id, api_hash)
        ClientPrivateMessageHandlerMixin.__init__(self)
        ClientGroupMessageHandlerMixin.__init__(self)
        self.query_manager = QueryManager()
