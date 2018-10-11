from __future__ import print_function

import random
import pandas as pd

root_dir = '/Users/andrecavalcante/Onedrive/alunos/van/'
data_csv = '/Users/andrecavalcante/Onedrive/alunos/van/data.csv'
     
def main():
    
    val_rate  = 0.2
    test_rate = 0.0
    
    data = pd.read_csv(data_csv)
    data_len    = len(data)
    
    val_len     = int(data_len * val_rate)
    test_len    = int(data_len * test_rate)

    data_idx = random.sample(range(data_len), data_len)
    
    val = data.loc[data_idx[:val_len]]
    test = data.loc[data_idx[val_len:val_len+test_len]]
    train = data.loc[data_idx[val_len:]]
    

    val.to_csv(root_dir + 'val.csv')
    test.to_csv(root_dir + 'test.csv')
    train.to_csv(root_dir + 'train.csv')

if __name__ == '__main__':
    main()
