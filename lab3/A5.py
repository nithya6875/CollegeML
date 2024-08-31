'''
A5. Train a kNN classifier (k =3) using the training set obtained from above exercise. Following code
for help:
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

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


