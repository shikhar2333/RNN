import torch
import torch.nn as nn

class LSTM(nn.Module):
    '''
    Returns the LSTM model object
    '''
    def __init__(self, input_features, hidden_features):
        super().__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        # i_t = sigma(U_i*x_t + V_i*h_t-1 + b_i)
        # Shape of x_t: (batch_size, sequence_size, input_features) => Shape of U_i
        # = (input_features, hidden_size)
        self.U_i = nn.Parameter(torch.Tensor(input_features, hidden_features))
        self.V_i = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
        self.b_i = nn.Parameter(torch.Tensor(hidden_features))

        # f_t = sigma(U_f*x_t + V_f*h_t-1 + b_f)
        self.U_f = nn.Parameter(torch.Tensor(input_features, hidden_features))
        self.V_f = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
        self.b_f = nn.Parameter(torch.Tensor(hidden_features))

        # g_t = tanh(U_g*x_t + V_g*h_t-1 + b_g)
        self.U_g = nn.Parameter(torch.Tensor(input_features, hidden_features))
        self.V_g = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
        self.b_g = nn.Parameter(torch.Tensor(hidden_features))

        # o_t = sigma(U_o*x_t + V_o*h_t-1 + b_o)
        self.U_o = nn.Parameter(torch.Tensor(input_features, hidden_features))
        self.V_o = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
        self.b_o = nn.Parameter(torch.Tensor(hidden_features))
        pass

    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        h_t, c_t = torch.zeros((batch_size, self.hidden_features)), torch.zeros((batch_size, self.hidden_features))
        for t in range(sequence_size):
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t@self.U_i + h_t@self.V_i + self.b_i)
            g_t = torch.tanh(x_t@self.U_g + h_t@self.V_g + self.b_g)
            f_t = torch.sigmoid(x_t@self.U_f + h_t@self.V_f + self.b_f)
            # c_t = c_t-1*f_t + i_t*g_t
            c_t = f_t*c_t + i_t*g_t
            # h_t = o_t*tanh(c_t)
            o_t = torch.tanh(x_t@self.U_o + h_t@self.V_o + self.b_o)
            h_t = o_t*torch.tanh(c_t)
        
        pass




