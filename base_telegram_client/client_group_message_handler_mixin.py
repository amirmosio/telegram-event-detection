import telethon
from conversation_models.neural_network.model_training import (
    ConversationRootModel,
)
import torch
from telethon.tl.types import PeerUser
from conversation_models.random_forest.data_preparation import (
    generate_dataset_from_labeled_data_with_sliding_window,
)
import joblib
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertForSequenceClassification,
)
import pandas as pd
from user_telegram_client.constants import TopicClasses
from conversation_models.random_forest.model_training import (
    draw_variable_affects_for_a_sample,
)


class ClientGroupMessageHandlerMixin:
    def __init__(self) -> None:

        self.rf_model = joblib.load("./trained_models/RF_93.joblib")
        self.embedding_distil_bert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.embedding_distil_bert_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.conversation_model_nn = ConversationRootModel(input_feature_size=768)
        self.conversation_model_nn.load_model(
            r"conversation_model_1715021680.755679_epoch_24_acc_83"
        )
        self.conversation_model_nn.eval()

        self.topic_distil_tokenizer = DistilBertTokenizer.from_pretrained(
            r"trained_models\DistilBERT\distilBERT_tokenizer"
        )

        # Load the DistilBERT model
        self.topic_distil_model = DistilBertForSequenceClassification.from_pretrained(
            r"trained_models\DistilBERT\distilBERT_model"
        )

        # Set the model in evaluation mode
        self.topic_distil_model.eval()

        @self.client.on(telethon.events.NewMessage(func=lambda e: e.is_group))
        async def group_message_handler(event):
            chat = await event.get_chat()

            if event.message.reply_to_msg_id:
                pass
            else:
                is_new_conv = await self.__check_if_it_is_a_new_conversation(
                    event, chat
                )
                if is_new_conv:
                    topic = self.__detect_topic_of_message(event.raw_text)
                    await self.__send_notif_to_users(event, chat, topic)

    #### Private Functions
    def __detect_topic_of_message(self, text):
        topic_map = {
            0: "1",
            1: "2",
            2: "4",
            3: "5",
            4: "6",
            5: "7",
            6: "8",
        }
        inputs = self.topic_distil_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.topic_distil_model(**inputs)
            logits = outputs.logits

        # Achieve Predictions
        prediction = torch.argmax(logits, dim=-1)
        topic = topic_map[prediction.item()]
        return topic

    async def __check_if_it_is_a_new_conversation(self, event, chat):
        messages = await self.get_messages_without_takeout(
            chat_id=chat.id, limit=5, offset_id=event.message.id + 1
        )
        messages = messages.iloc[::-1]
        messages = messages.reset_index(
            drop=True
        )  # Performing [::-1] does not reset index :)
        input_for_rf_df = generate_dataset_from_labeled_data_with_sliding_window(
            messages,
            window_size=4,
            label_column=False,
            embedding_model=self.embedding_distil_bert_model,
            embedding_tokenizer=self.embedding_distil_bert_tokenizer,
            conversation_model_nn=self.conversation_model_nn,
        )
        is_related = self.rf_model.predict(input_for_rf_df)[-1]
        from main import DEBUG

        if DEBUG:
            y = y = pd.DataFrame([])
            y = y._append({"y": is_related}, ignore_index=True)["y"]
            draw_variable_affects_for_a_sample(
                self.rf_model, input_for_rf_df, y, n=1, title_extra=event.raw_text
            )
        return not is_related

    async def __send_notif_to_users(self, event, chat, topic=None):
        users = self.query_manager.get_users_interested_in_topic_and_group(
            group=chat.id, topic=topic
        )
        for user in users:
            user_peer = PeerUser(int(user))
            await self.client.send_message(
                user_peer,
                f"{chat.title} - {TopicClasses[int(topic)]}:\n{event.raw_text}",
            )
