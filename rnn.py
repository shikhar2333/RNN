import numpy as np
from dataloader import one_hot_encoded_input
from numpy.random import randn
# sigmoid function
def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))
# Derivative of sigmoid function
def sigmoid_prime(sigmoid_z):
  return sigmoid_z * (1-sigmoid_z)


class BasicRNN:
  '''
  returns the basic RNN model object( many to one RNN )
  Shape of x_t = (vocab_size, 1)
  Shape of W_xh = (hidden_size, vocab_size)
  Shape of W_hh = (hidden_size, hidden_size)
  Shape of W_hy = (output_features, hidden_size)
  Shape of b_h = (hidden_size, 1)
  h_t = W_xh@x_t + W_hh@h_t-1 + b_h
  y_n = W_hy@h_n + b_y
  '''
  def __init__(self, input_features, output_features, hiddden_size = 64):
    self.input_features = input_features
    self.output_features = output_features
    self.hiddden_size = hiddden_size
    '''
    Weight Matrices
    '''
    self.W_xh = randn(hiddden_size, input_features)
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
    h_t = np.zeros((self.hiddden_size, 1))
    # print(h_t.shape)
    self.x_t = []
    self.h_t = [h_t]
    # print("Sentence: ", x)
    for word in x.split():
      x_t = one_hot_encoded_input(word)
      self.x_t.append(x_t)
      a_t = self.W_xh@x_t + self.W_hh@h_t + self.b_h
      h_t = sigmoid(a_t)
      self.h_t.append(h_t)
    
    # print(self.W_hy.shape, h_t.shape)
    y = self.W_hy@h_t + self.b_y
    return y

  def backward(self, p, lr = 0.01):
    '''
    implements the BPPT algorithm
    L = -ln(softmax(y_correct))
    softmax(y_i) = e^y_i/sum(e^y_i)
    dL/dy_i = p_i if i not correct else 1 - p_i
    dL/db_y = dL/dy@dy/db_y = dL/dy 
    dL/dW_hy = dL/dy@dy/dW_hy = dL/dy@h_n
    dL/dh_t = dL/dh_t+1@d_ht+1/dh_t
    dL/db_h = dL/dy@sum(dy/dh_t@dh_t/db_h)
    dL/dW_xh = dL/dy@sum(dy/dh_t@dh_t/dW_xh)
    dL/dW_hh = sum(dL/dh_t@dh_t/dW_hh)
    dh_t/dW_hh = sigmoid_prime(a_t)@h_t-1
    dy/dh_t = dy/dh_t+1@d_ht+1/dh_t
    dh_t+1/dh_t = sigmoid_prime(a_t)@W_hh
    '''
    dL_dy, dL_db_y, dL_dW_hy = p, p, p@self.h_t[-1].T
    t = len(self.x_t)
    incoming_gradient = dL_dy
    dL_db_h, dL_dW_xh, dL_dW_hh = np.zeros(self.b_h.shape), np.zeros(self.W_xh.shape), np.zeros(self.W_hh.shape)
    # Computes dL/dh_t with dL/dh_n = dL/dy@W_hy
    dL_dh_n = dL_dy.T@self.W_hy
    layer_wise_loss_grads = dL_dh_n.T
    n = len(self.x_t)
    for i in range(n - 1, -1, -1):
      # print("dL/dh_n shape", layer_wise_loss_grads.shape, sigmoid_prime(self.h_t[i]).shape, self.x_t[i].shape)
      dL_db_h += layer_wise_loss_grads*sigmoid_prime(self.h_t[i + 1])
      dL_dW_xh += layer_wise_loss_grads*sigmoid_prime(self.h_t[i + 1])@self.x_t[i].T
      dL_dW_hh += layer_wise_loss_grads*sigmoid_prime(self.h_t[i + 1])@self.h_t[i].T
      layer_wise_loss_grads = self.W_hh@layer_wise_loss_grads*sigmoid_prime(self.h_t[i + 1])
    
    # gradient update eqn's
    self.b_y -= lr*dL_db_y
    self.W_hy -= lr*dL_dW_hy
    
    self.b_h -= lr*dL_db_h
    self.W_xh -= lr*dL_dW_xh
    self.W_hh -= lr*dL_dW_hh
