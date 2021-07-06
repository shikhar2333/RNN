import numpy as np
from collections import defaultdict
from data import train_data
vocab_dict = defaultdict(int)
for sentence in train_data.keys():
  for word in sentence.split():
    vocab_dict[word] += 1

print("Length of Vocabulary: ", len(vocab_dict.keys()))
vocab_size = len(vocab_dict.keys())

def one_hot_encoded_input(sentence):
  '''
  Returns the one hot encoded reperesentation of the sentence
  '''
  
  word_index = {word: i for i, word in enumerate(vocab_dict)}
  one_hot_sentence = []
  for i, word in enumerate(sentence.split()):
    index = word_index[word]
    zeros = np.zeros(vocab_size)
    zeros[index] = 1
    one_hot_sentence.append(zeros)

  return np.asarray(one_hot_sentence)