from transformers import (
    RobertaTokenizer,
    RobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
)
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def embedding_with_sentence_transformer(messages):
    emodel = SentenceTransformer("average_word_embeddings_glove.6B.300d")  # 300
    return list(emodel.encode(messages))


# def embedding_with_laser(messages):
#     from laserembeddings import Laser

#     laser = Laser()
#     return laser.embed_sentences(messages, lang=["en"] * len(len(messages)))


def embedding_with_distil_bert(messages, tokenizer=None, model=None):
    tokenizer = (
        tokenizer
        if tokenizer
        else DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    )
    model = (
        model if model else DistilBertModel.from_pretrained("distilbert-base-uncased")
    )

    def calculate_embedding(m):
        input_tokens = tokenizer(
            m,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        return model(**input_tokens)

    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(messages))):
            try:
                output = calculate_embedding(messages[i])
            except:
                print("Error on embedding")
                output = calculate_embedding("")

            embeddings.append(output.last_hidden_state[:, 0, :].flatten())
    return embeddings
