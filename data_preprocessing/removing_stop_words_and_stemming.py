import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
# more ignoring this preprecessing for now
def remove_stop_words(df_original, column="text"):
    texts = df_original[column]
    df_result = df_original.copy()
    text_tokens = [word_tokenize(text) for text in texts ]
    df_result['text_without_stop_words'] = [" ".join(m) for m in text_tokens]
    return df_result

def stemming(df_original):
    # TODO
    return df_original