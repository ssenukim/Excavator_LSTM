import torch 
import torch.nn as nn 
from torch import optim 
import torch.nn.functional as F 
import numpy as np 
import random 
import os, errno 
import sys 
from tqdm import trange
from Utility import *


class LSTM_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                           num_layers = num_layers, bidirectional=False, batch_first=True)
    def forward(self, x_input):
        hidden = (torch.zeros(self.num_layers, x_input.shape[0], self.hidden_size).to('cuda:0'), torch.zeros(self.num_layers, x_input.shape[0], self.hidden_size).to('cuda:0'))                     
        
        lstm_out, self.hidden = self.lstm(x_input, hidden)
        return lstm_out, self.hidden 
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
               torch.zeros(self.num_layers, batch_size, self.hidden_size))

class LSTM_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2=5, num_layers=1):
        super(LSTM_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                           num_layers = num_layers, batch_first = True)
        self.linear_1 = nn.Linear(hidden_size, hidden_size_2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(hidden_size_2, input_size)
        
    def forward(self, x_input, encoder_hidden_states):
        lstm_out, hidden = self.lstm(x_input.clone(), encoder_hidden_states)
        lstm_out = lstm_out.squeeze(1)
        output = self.linear_1(lstm_out)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear_2(output)
        return output
    
class LSTM_seq2seq(nn.Module):
    def __init__(self, input_size_enc, input_size_dec, hidden_size, fc_hidden_size, num_layers=1):
        super(LSTM_seq2seq, self).__init__()
        self.input_size_enc = input_size_enc
        self.input_size_dec = input_size_dec 
        self.hidden_size = hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.num_layers = num_layers
        
        self.encoder = LSTM_encoder(input_size=input_size_enc, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = LSTM_decoder(input_size=input_size_dec, hidden_size=hidden_size,
                                    hidden_size_2=fc_hidden_size, num_layers=num_layers)
        
    def forward(self, inputs, targets, decoder_input, target_len=3, teacher_forcing_ratio=0.6):
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        decoder_input = decoder_input
        
        outputs = torch.zeros(batch_size, target_len)
        
        _, hidden = self.encoder(inputs)
        
        for t in range(target_len):
            out = self.decoder(decoder_input, hidden)
            decoder_input = decoder_input.squeeze()
            if random.random() < teacher_forcing_ratio:
                decoder_input[:, t] = targets[:, t]
            else:
                decoder_input[:, t] = out[:, t]
            decoder_input = decoder_input.unsqueeze(1)
            #print(type(out), out.shape, decoder_input.shape)
            outputs[:, t] = out[:, t]
           
        return outputs
    
    def get_value(self, inputs, decoder_input, target_len=3):
        self.eval()
        batch_size = inputs.shape[0]
        decoder_input = decoder_input
        
        outputs = torch.zeros(batch_size, target_len)
        
        _, hidden = self.encoder(inputs)
        
        for t in range(target_len):
            out = self.decoder(decoder_input, hidden)
            decoder_input = torch.squeeze(decoder_input, 1)
            decoder_input[:, t] = out[:, t]
            decoder_input = decoder_input.unsqueeze(1)
            outputs[:, t] = out[:, t]
        outputs = outputs.detach().numpy()
        return outputs    
        
    def generate(self, inputs, decoder_input, stop_angle=0.18, target_len=3):
        self.eval()
        inputs = inputs
        decoder_input = decoder_input 
        result = np.zeros((1, 3))
        batch_size = inputs.shape[0]
        
        
        while result[-1, 2] < stop_angle:
            outputs = torch.zeros(batch_size, target_len)
            _, hidden = self.encoder(inputs)
            for t in range(target_len):
                out = self.decoder(decoder_input, hidden)
                decoder_input = torch.squeeze(decoder_input, 1)
                decoder_input[:, t] = out[:, t]
                decoder_input = decoder_input.unsqueeze(1)
                outputs[:, t] = out[:, t]
            outputs = outputs.detach().numpy()
            result = np.concatenate((result, outputs), axis=0)
            
            inputs[:, :, 6] = inputs[:, :, 6] + 0.01 
        
        result = np.delete(result, 0, axis=0)
        return result
    
    def save(self, PATH, state_dict=True):
        if state_dict==True:
            torch.save(self.state_dict(), PATH)
        else:
            torch.save(self, PATH)
        return
            
        
        
           
      