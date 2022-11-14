import torch 
from torch.utils.data import DataLoader, Dataset 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from LSTM_model import *
from Utility import *

model = LSTM_seq2seq(7, 3, 15, 64).to('cuda:0')
model.load_state_dict(torch.load('C:/Users/User/Programming/Excavator/LSTM/Trained_model/model_1.pt'))

test_input_list = make_input_tensor('C:/Users/USER/Programming/Excavator/Data/Excavation_test_data_refrained.csv', 0, 4)
decoder_input = torch.FloatTensor(1, 1, 3).fill_(-1)
decoder_input = decoder_input.to('cuda:0')

result_list = []
for inputs in test_input_list:
    result = model.generate(inputs, decoder_input, 0.21)
    result_list.append(result)

test_dataset_answer_list = []
for n in range(4):
    df = pd.read_csv('C:/Users/USER/Programming/Excavator/Data/Excavation_test_answer_' + str(n+1) + '.csv')
    test_dataset_answer = torch.FloatTensor(df.values)
    test_dataset_answer = test_dataset_answer.numpy()
    test_dataset_answer_list.append(test_dataset_answer) 
    
model_score = total_MAPE_score(result_list, test_dataset_answer_list)
print('total score: ', model_score)
