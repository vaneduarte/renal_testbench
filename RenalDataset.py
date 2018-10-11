from __future__ import print_function


import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

class RenalDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = {
            'sexo': torch.tensor(float(self.data.loc[idx, 'sexo'])).unsqueeze(0), 
            'idade': torch.tensor(float(self.data.loc[idx, 'idade'])).unsqueeze(0), 
            'altura': torch.tensor(float(str(self.data.loc[idx, 'altura ']).replace(',','.'))).unsqueeze(0), 
            'peso': torch.tensor(float(str(self.data.loc[idx, 'Peso ']).replace(',','.'))).unsqueeze(0), 
            'imc': torch.tensor(float(str(self.data.loc[idx, 'IMC']).replace(',','.'))).unsqueeze(0), 
            'pas': torch.tensor(float(str(self.data.loc[idx, 'PAS']).replace(',','.'))).unsqueeze(0), 
            'pad': torch.tensor(float(str(self.data.loc[idx, 'PAD']).replace(',','.'))).unsqueeze(0), 
            'cc': torch.tensor(float(str(self.data.loc[idx, 'CC']).replace(',','.'))).unsqueeze(0),
            'glicose': torch.tensor(float(str(self.data.loc[idx, 'Glicose']).replace(',','.'))).unsqueeze(0), 
            'creatinina': torch.tensor(float(str(self.data.loc[idx, 'creatinina']).replace(',','.'))).unsqueeze(0), 
            'colesterol': torch.tensor(float(str(self.data.loc[idx, 'Colesterol total']).replace(',','.'))).unsqueeze(0), 
            'hdl': torch.tensor(float(str(self.data.loc[idx, 'HDL']).replace(',','.'))).unsqueeze(0),
            'ldl': torch.tensor(float(str(self.data.loc[idx, 'LDL']).replace(',','.'))).unsqueeze(0), 
            'trig': torch.tensor(float(str(self.data.loc[idx, 'Triglicerideos']).replace(',','.'))).unsqueeze(0), 
            #'txa': torch.tensor(float(self.data.loc[idx, 'txa'].replace(',','.'))).unsqueeze(0), 
            'rotulo': torch.tensor(float(self.data.loc[idx, 'Rotulagem'])).long()}
        
        return sample

