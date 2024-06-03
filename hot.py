from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Example datasets
words_dataset = ["hello world", "machine learning"]
characters_dataset = ["abc", "def"]

# Splitting the datasets
words = [word for sentence in words_dataset for word in sentence.split()]
characters = [char for string in characters_dataset for char in string]

# Unique elements
unique_words = list(set(words))
unique_characters = list(set(characters))

# Reshaping the data to fit OneHotEncoder input requirements
words_reshaped = np.array(words).reshape(-1, 1)
characters_reshaped = np.array(characters).reshape(-1, 1)

# One-hot encoding
encoder_words = OneHotEncoder(sparse=False)
encoder_characters = OneHotEncoder(sparse=False)

one_hot_words = encoder_words.fit_transform(words_reshaped)
one_hot_characters = encoder_characters.fit_transform(characters_reshaped)

# Displaying the result
(one_hot_words, one_hot_characters)
