
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
print("Matplotlib is installed correctly!")

# Example dataset: List of social media posts (tweets or comments)
data = [
    "I love this product! It's amazing and really useful.",
    "Worst experience ever! Totally disappointed.",
    "Not sure how I feel about this, it's okay I guess.",
    "This movie was fantastic! Highly recommend it.",
    "I hate the customer service. Never coming back.",
    "The weather is nice today, but still i am not in good mood''
]

# Labels: 1 = Positive, 0 = Negative, 2 = Neutral
labels = [1, 0, 2, 1, 0, 2]
def preprocess_text(text):
    """
    Preprocess the text data: remove special characters, lowercasing, etc.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Preprocess all posts
preprocessed_data = [preprocess_text(post) for post in data]
print(preprocessed_data)
def tokenize(text):
    """
    Tokenize the text into individual words.
    """
    return text.split()

def build_vocab(data):
    """
    Build a vocabulary (word frequency count).
    """
    all_words = []
    for text in data:
        all_words.extend(tokenize(text))
    
    word_count = Counter(all_words)
    return word_count

# Build vocabulary
vocab = build_vocab(preprocessed_data)
print(vocab)
def vectorize_text(text, vocab):
    """
    Convert a text post into a vector of word frequencies based on the vocabulary.
    """
    vector = np.zeros(len(vocab))
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    for word in tokenize(text):
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    
    return vector

# Create the feature vectors for all posts
vocab = list(vocab)
vectors = np.array([vectorize_text(post, vocab) for post in preprocessed_data])
print(vectors)
def sentiment_classifier(text_vector, vocab):
    """
    Simple sentiment classifier based on positive/negative word counts.
    """
    positive_words = ['love', 'amazing', 'fantastic', 'good', 'recommend']
    negative_words = ['hate', 'worst', 'disappointed', 'terrible', 'never']

    pos_count = sum(text_vector[vocab.index(word)] for word in positive_words if word in vocab)
    neg_count = sum(text_vector[vocab.index(word)] for word in negative_words if word in vocab)

    if pos_count > neg_count:
        return 1  # Positive
    elif neg_count > pos_count:
        return 0  # Negative
    else:
        return 2  # Neutral

# Classify the sentiment of each post
predictions = [sentiment_classifier(vector, vocab) for vector in vectors]
print(predictions)
def accuracy(predictions, labels):
    """
    Calculate the accuracy of the sentiment predictions.
    """
    return np.mean(np.array(predictions) == np.array(labels))

# Evaluate accuracy
acc = accuracy(predictions, labels)
print(f"Accuracy: {acc * 100:.2f}%")
def plot_sentiment_distribution(labels, predictions):
    """
    Visualize the distribution of sentiment predictions.
    """
    plt.figure(figsize=(8, 5))
    plt.hist([labels, predictions], bins=3, label=['True Labels', 'Predictions'], alpha=0.7)
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Sentiment Distribution')
    plt.show()

# Plot the sentiment distribution
plot_sentiment_distribution(labels, predictions)
