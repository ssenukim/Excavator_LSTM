import torch 
from torch.utils.data import DataLoader, Dataset 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from LSTM_model import *


if __name__ == '__main__':    
    CSV_PATH = 'C:/Users/USER/Programming/Excavator/Data/totalExcavationDataR4_cut.csv'

    train_dataset_1 = CustomDataset(CSV_PATH, 0, 160000)
    #train_dataset_2 = CustomDataset(CSV_PATH, 20000, 150000)
    train_dataloader_1 = DataLoader(train_dataset_1, batch_size=32, shuffle=True, num_workers=2)
    #train_dataloader_2 = DataLoader(train_dataset_2, batch_size=32, shuffle=True, num_workers=2)


    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name(0))

    model = LSTM_seq2seq(input_size_enc=7, input_size_dec=3, hidden_size=15, fc_hidden_size=64).to(device)

    learning_rate = 0.0005
    num_epoch = 120
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()

    from tqdm import tqdm


    model.train()
    with tqdm(range(num_epoch)) as tr:
        for i in tr:
            total_loss = 0.0
            for x, y in train_dataloader_1:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
                #decoder_input = torch.zeros(64, 1, 3)
                decoder_input_1 = torch.FloatTensor(32, 1, 3).fill_(-1)
                decoder_input_1 = decoder_input_1.to(device)
           
                output = model.forward(x, y, decoder_input_1)
                output = output.to(device)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_dataloader_1)))
    '''    
    with tqdm(range(num_epoch)) as tr:
        for i in tr:
            total_loss = 0.0
            for x, y in train_dataloader_2:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
                #decoder_input = torch.zeros(64, 1, 3)
                decoder_input_2 = torch.FloatTensor(32, 1, 3).fill_(-1)
                decoder_input_2 = decoder_input_2.to(device)
           
                output = model.forward(x, y, decoder_input_2)
                output = output.to(device)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_dataloader_2)))
        '''
    model.save('C:/Users/USER/Programming/Excavator/LSTM/Trained_model/' +  'model_1.pt', state_dict=True)
    print()
    print('finish')