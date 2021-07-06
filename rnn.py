import numpy as np
from numpy.random import randn

# sigmoid function
def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))
# Derivative of sigmoid function
def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

class BasicRNN:
  '''
  returns the basic RNN model object( many to one RNN )
  Shape of x_t = (vocab_size, 1)
  Shape of W_xh = (hidden_size, vocab_size)
  Shape of W_hh = (hidden_size, hidden_size)
  Shape of W_hy = (output_features, hidden_size)
  Shape of b_h = (hidden_size, 1)
  h_t = W_xh@x_t + W_hh@h_t-1 + b_h
  y_t = W_hy@h_t + b_y
  '''
  def __init__(self, input_features, output_features, hiddden_size = 64):
    self.input_features = input_features
    self.output_features = output_features
    self.hiddden_size = hiddden_size
    '''
    Weight Matrices
    '''
    self.W_xh = randn(input_features, hiddden_size)
    self.W_hh = randn(hiddden_size, hiddden_size)
    self.W_hy = randn(output_features, hiddden_size)
    '''
    Biases 
    '''
    self.b_h = randn(hiddden_size, 1)
    self.b_y = randn(output_features, 1)
  

  def forward(self, x):
    '''
    computes the forward pass for the RNN model
    '''
    h = np.zeros((self.hiddden_size, 1))
    self.h_t = [h]
    for t, one_hot_encoded_word in enumerate(x):
      x_t = one_hot_encoded_word
      a_t = self.W_xh@x_t + self.W_hh@self.h_t[-1] + self.b_h
      self.h_t.append([sigmoid(z) for z in a_t])
    
    y = self.W_hy@self.h_t[-1] + self.b_y
    return self.h_t[-1], y

  def backward(self):
    '''
    implements the BPPT algorithm
    L = -ln(softmax(y_correct))
    softmax(y_i) = e^y_i/sum(e^y_i)
    dL/dy_i = p_i if i not correct else 1 - p_i
    dL/db_y = dL/dy@dy/db_y = dL/dy
    dL/dW_hy = dL/dy@dy/dW_hy = dL/dy@h_t
    dL/db_h = dL/dy@sum(dy/dh_t@dh_t/db_h)
    dL/dW_xh = dL/dy@sum(dy/dh_t@dh_t/dW_xh)
    '''