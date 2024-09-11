import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from scipy.sparse import hstack, csr_matrix

def metrics(labels, labels_pred):
    # Performance metrics calculation for multiclass
    precision_train = precision_score(labels, labels_pred, average='weighted')
    recall_train = recall_score(labels, labels_pred, average='weighted')
    f1_train = f1_score(labels, labels_pred, average='weighted')
    return precision_train, recall_train, f1_train

def kNN_classifier(features, labels):
    # kNN classifier
    model = KNeighborsClassifier()
    model.fit(features, labels)
    return model.predict(features)

def main():
    sentence_data = pd.read_csv('lab3_labelled_data.csv')

    # TF-IDF vectorization for text feature extraction
    tfidf_vectorizer = TfidfVectorizer()
    text = tfidf_vectorizer.fit_transform(sentence_data["Text"])

    # Numerical data of the dataset (converted to sparse matrix)
    numerical_data = sentence_data.iloc[:, 1:-1]

    # Ensure all columns in numerical_data are numeric
    numerical_data = numerical_data.apply(pd.to_numeric, errors='coerce')

    # Handle any missing values that might result from conversion
    numerical_data = numerical_data.fillna(0)

    # Convert numerical data to a sparse matrix
    numerical_data = csr_matrix(numerical_data.values)

    # Labels of the dataset
    labels = sentence_data['label'].values

    # Combine text features with numerical data
    combined_features = hstack([text, numerical_data])

    # Convert sparse matrix to dense because kNN does not accept sparse matrices
    combined_features = combined_features.toarray()

    # Split the data into training and testing sets
    training_features, testing_features, training_labels, testing_labels = train_test_split(combined_features, labels, test_size=0.3, random_state=42)

    # Prediction on training and testing
    training_labels_pred = kNN_classifier(training_features, training_labels)
    testing_labels_pred = kNN_classifier(testing_features, testing_labels)

    # Confusion matrix
    confusion_train = confusion_matrix(training_labels, training_labels_pred)
    confusion_test = confusion_matrix(testing_labels, testing_labels_pred)

    print("CONFUSION MATRIX OF TRAINING DATA:")
    print(confusion_train)

    # Metrics printing
    precision_train, recall_train, f1_train = metrics(training_labels, training_labels_pred)
    print("PERFORMANCE METRICS:")
    print(f"PRECISION: {precision_train:.2f}")
    print(f"RECALL: {recall_train:.2f}")
    print(f"F1-SCORE: {f1_train:.2f}")
    print(" ")

    print("CONFUSION MATRIX OF TESTING DATA:")
    print(confusion_test)

    precision_test, recall_test, f1_test = metrics(testing_labels, testing_labels_pred)
    print("PERFORMANCE METRICS:")
    print(f"PRECISION: {precision_test:.2f}")
    print(f"RECALL: {recall_test:.2f}")
    print(f"F1-SCORE: {f1_test:.2f}")

if __name__ == "__main__":
    main()

