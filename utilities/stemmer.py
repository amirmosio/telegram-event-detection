from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

assert nltk.download("punkt")


def stem_and_tokenize_the_sentence(message):
    ps = PorterStemmer()
    words = word_tokenize(message)
    stemmed_words = [ps.stem(w) for w in words]
    return stemmed_words
