from sklearn import preprocessing
from tqdm import tqdm
from utilities.stemmer import stem_and_tokenize_the_sentence
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def get_categories_searching_through_keywrods(messages, category_keyword_map):
    categories = []
    for message in tqdm(messages):

        categories_matches = {}
        for key in category_keyword_map:
            keywords = category_keyword_map[key]
            matches = [
                k
                for k in keywords
                if all(
                    [
                        token in stem_and_tokenize_the_sentence(message)
                        for token in stem_and_tokenize_the_sentence(k)
                    ]
                )
            ]
            categories_matches[key] = len(matches)

        if max(categories_matches.values()) != 0:
            categories.append(max(categories_matches, key=categories_matches.get))
        else:
            categories.append("8. Other")
    return categories


def print_accuracy(predicted_topics, real_topics):

    # Create a label (category) encoder object
    le = preprocessing.LabelEncoder()

    # Fit the encoder to the true labels
    le.fit(real_topics)

    # Apply the fitted encoder to the true and predicted labels
    true_encoded = le.transform(real_topics)
    predicted_encoded = le.transform(predicted_topics)

    print("CM:")
    print(confusion_matrix(true_encoded, predicted_encoded))

    overall_precision = precision_score(
        true_encoded, predicted_encoded, average="macro"
    )
    overall_recall = recall_score(true_encoded, predicted_encoded, average="macro")
    overall_f1 = f1_score(true_encoded, predicted_encoded, average="macro")

    class_report = classification_report(true_encoded, predicted_encoded)

    print(f"Overall Precision: {overall_precision}")
    print(f"Overall Recall: {overall_recall}")
    print(f"Overall F1-score: {overall_f1}")

    print("Classification Report:")
    print(class_report)
