import telethon
from base_telegram_client.client import TelegramClient
from base_telegram_client.client_download_message_mixin import (
    ClientDownloadMessageMixin,
)


class DataCollectorClient(TelegramClient, ClientDownloadMessageMixin):

    def __init__(self, phone_number, api_id, api_hash):
        super(DataCollectorClient, self).__init__(phone_number, api_id, api_hash)
