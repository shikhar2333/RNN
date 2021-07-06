import numpy as np
from rnn import BasicRNN
from dataloader import one_hot_encoded_input, vocab_size
from data import train_data, test_data
no_epochs = 500

def softmax(y):
    return np.exp(y)/sum(np.exp(y))

rnn_model = BasicRNN(vocab_size, 2)
for epoch in range(no_epochs):
    # Run the forward pass
    sentence_train = train_data.items()
    for sentence, label in sentence_train:
        h, y = rnn_model.forward(sentence)
        probabilities = softmax(y)
        loss = -np.log(probabilities[label])