import numpy as np
from rnn import BasicRNN, sigmoid
from dataloader import vocab_size
from data import train_data, test_data
import matplotlib.pyplot as plt

no_epochs = 1000


def softmax(y):
    return np.exp(y) / sum(np.exp(y))


rnn_model = BasicRNN(vocab_size, 2)
losses = []
iters = []
test_loss = []
for epoch in range(no_epochs):
    sentence_train = train_data.items()
    loss = 0
    for sentence, label in sentence_train:
        y = rnn_model.forward(sentence)
        probabilities = softmax(y).T
        label = 1 if label else 0
        loss -= np.log(probabilities[0][label])
        probabilities[0][label] -= 1
        rnn_model.backward(probabilities.T)

    if epoch % 10:
        losses.append(loss)
        iters.append(epoch)
        sentence_test = test_data.items()
        loss = 0
        for sentence, label in sentence_test:
            y = rnn_model.forward(sentence)
            probabilities = softmax(y).T
            label = 1 if label else 0
            loss -= np.log(probabilities[0][label])
        print("Testing loss: ", loss)
        test_loss.append(loss)


plt.plot(iters, losses)
plt.ylabel("Training Loss")
plt.xlabel("epoch number")
# plt.savefig("RNN_Train_Loss.png")
# plt.show()
plt.plot(iters, test_loss)
plt.ylabel("Test Loss")
plt.xlabel("epoch number")
plt.legend(["Train Loss", "Test Loss"], loc="upper right")
plt.savefig("RNN_Losses.png")
# plt.show()
