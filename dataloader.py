import numpy as np
from collections import defaultdict
from data import train_data

vocab_dict = defaultdict(int)
for sentence in train_data.keys():
    for word in sentence.split():
        vocab_dict[word] += 1

print("Length of Vocabulary: ", len(vocab_dict.keys()))
vocab_size = len(vocab_dict.keys())
word_index = {word: i for i, word in enumerate(vocab_dict)}


def one_hot_encoded_input(word):
    """
  Returns the one hot encoded reperesentation of the word
  """
    one_hot_sentence = np.zeros((vocab_size, 1))
    index = word_index[word]
    one_hot_sentence[index] = 1
    return one_hot_sentence
