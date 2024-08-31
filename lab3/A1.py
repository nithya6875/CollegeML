import pandas as pd
import numpy as np

# Load your dataset
file_path = 'lab3_labelled_data.csv'
df = pd.read_csv(file_path)

# Separate the dataset into questions and answers
questions = df[df['label'] == 'question']['Text']
answers = df[df['label'] == 'answer']['Text']
from sklearn.feature_extraction.text import CountVectorizer

# Vectorize the text data
vectorizer = CountVectorizer()
questions_vectors = vectorizer.fit_transform(questions).toarray()
answers_vectors = vectorizer.transform(answers).toarray()
# Calculate the mean vector for each class
centroid_questions = np.mean(questions_vectors, axis=0)
centroid_answers = np.mean(answers_vectors, axis=0)

print("Centroid for Questions:", centroid_questions)
print("Centroid for Answers:", centroid_answers)
# Calculate the standard deviation for each class
spread_questions = np.std(questions_vectors, axis=0)
spread_answers = np.std(answers_vectors, axis=0)

print("Spread for Questions:", spread_questions)
print("Spread for Answers:", spread_answers)
# Calculate the Euclidean distance between the centroids
distance = np.linalg.norm(centroid_questions - centroid_answers)
print(f"Distance between the centroids: {distance:.2f}")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensionality to 2D
pca = PCA(n_components=2)
questions_pca = pca.fit_transform(questions_vectors)
answers_pca = pca.transform(answers_vectors)

# Calculate new centroids in 2D space
centroid_questions_2d = np.mean(questions_pca, axis=0)
centroid_answers_2d = np.mean(answers_pca, axis=0)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(questions_pca[:, 0], questions_pca[:, 1], color='blue', alpha=0.5, label='Questions')
plt.scatter(answers_pca[:, 0], answers_pca[:, 1], color='red', alpha=0.5, label='Answers')

# Plotting the centroids
plt.scatter(centroid_questions_2d[0], centroid_questions_2d[1], color='blue', marker='x', s=100, label='Questions Centroid')
plt.scatter(centroid_answers_2d[0], centroid_answers_2d[1], color='red', marker='x', s=100, label='Answers Centroid')

# Adding labels and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Questions and Answers')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()