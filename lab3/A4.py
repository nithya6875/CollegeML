import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

file_path = "lab3_labelled_data.csv"
df = pd.read_csv(file_path)

print(df.head)

# Features and labels
X = df['Text']  # The feature vector set, which in this case is the 'Text' column
y = df['label']  # The labels, which is the 'label' column

# Split the data into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shapes of the training and test sets
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

'''
A5. Train a kNN classifier (k =3) using the training set obtained from above exercise. Following code
for help:
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

'''

# Vectorize the text data (convert text into numeric features)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the kNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_vec, y_train)

# Evaluate the classifier on the test set
accuracy = neigh.score(X_test_vec, y_test)
print(f"Accuracy of kNN classifier on the test set: {accuracy:.2f}")
