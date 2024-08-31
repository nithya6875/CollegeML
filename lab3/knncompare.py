import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

file_path = "lab3_labelled_data.csv"
df = pd.read_csv(file_path)

# Load your dataset (assuming you already have the dataframe 'df' from your earlier steps)
X = df['Text']  # or the feature vectors if you've already vectorized the text
y = df['label']  # labels: 'question' or 'answer'

# Convert text data to feature vectors if not already done (e.g., using CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = range(1, 21)  # You can choose a range depending on your data
accuracies = []

for k in k_values:
    # Initialize the kNN classifier with the current k value
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Optional: Print classification report and confusion matrix for each k
    print(f"\nClassification Report for k={k}:\n")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix for k={k}:\n")
    print(confusion_matrix(y_test, y_pred))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title('k-NN Classifier Accuracy for Different k Values')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

