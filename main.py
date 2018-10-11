import torch
import torch.nn as nn
import torch.nn.functional as F

from RenalDataset import RenalDataset


train_csv = '/home/andre/alunos/van/train.csv'
val_csv = '/home/andre/alunos/van/val.csv'
test_dir = '/home/andre/alunos/van/test.csv'


input_size = 13
hidden_size = 100
num_classes = 2
num_epochs = 1000
batch_size = 1
learning_rate = 0.0005



train_dataset = RenalDataset(csv_file=train_csv)
val_dataset = RenalDataset(csv_file=val_csv)

train_loader = torch.utils.data.DataLoader(
                                   dataset=train_dataset,
                                   batch_size=batch_size, 
                                   shuffle=True,
                                   num_workers=8)

val_loader = torch.utils.data.DataLoader(
                                   dataset=val_dataset,
                                   batch_size=1, 
                                   num_workers=8)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/8))
        self.fc3 = nn.Linear(int(hidden_size/8), int(hidden_size/8))
        
        
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, samples in enumerate(train_loader):  
        sexo = samples['sexo']
        sexo[torch.isnan(sexo)] = 0.0
        idade = samples['idade']
        idade[torch.isnan(idade)] = 0.0
        altura = samples['altura']
        altura[torch.isnan(altura)] = 0.0
        peso = samples['peso']
        peso[torch.isnan(peso)] = 0.0
        imc = samples['imc']
        imc[torch.isnan(imc)] = 0.0
        pas = samples['pas']
        pas[torch.isnan(pas)] = 0.0
        pad = samples['pad']
        pad[torch.isnan(pad)] = 0.0
        cc = samples['cc']
        cc[torch.isnan(cc)] = 0.0
        glicose = samples['glicose']
        glicose[torch.isnan(glicose)] = 0.0
        creatinina = samples['creatinina']
        creatinina[torch.isnan(creatinina)] = 0.0
        colesterol = samples['colesterol']
        colesterol[torch.isnan(colesterol)] = 0.0
        hdl = samples['hdl']
        hdl[torch.isnan(hdl)] = 0.0
        ldl = samples['ldl']
        ldl[torch.isnan(ldl)] = 0.0
        trig = samples['trig']
        trig[torch.isnan(trig)] = 0.0
        rotulo = samples['rotulo']


        x = torch.cat([sexo, idade, altura, peso, imc, pas, pad, glicose, creatinina, colesterol, hdl, ldl, trig], 1)

        # Forward pass
        outputs = model(x)

        loss = criterion(outputs, rotulo)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for sample in val_loader:
        
        sexo = sample['sexo']
        sexo[torch.isnan(sexo)] = 0.0
        idade = sample['idade']
        idade[torch.isnan(idade)] = 0.0
        altura = sample['altura']
        altura[torch.isnan(altura)] = 0.0
        peso = sample['peso']
        peso[torch.isnan(peso)] = 0.0
        imc = sample['imc']
        imc[torch.isnan(imc)] = 0.0
        pas = sample['pas']
        pas[torch.isnan(pas)] = 0.0
        pad = sample['pad']
        pad[torch.isnan(pad)] = 0.0
        cc = sample['cc']
        cc[torch.isnan(cc)] = 0.0
        glicose = sample['glicose']
        glicose[torch.isnan(glicose)] = 0.0
        creatinina = sample['creatinina']
        creatinina[torch.isnan(creatinina)] = 0.0
        colesterol = sample['colesterol']
        colesterol[torch.isnan(colesterol)] = 0.0
        hdl = sample['hdl']
        hdl[torch.isnan(hdl)] = 0.0
        ldl = sample['ldl']
        ldl[torch.isnan(ldl)] = 0.0
        trig = sample['trig']
        trig[torch.isnan(trig)] = 0.0
        rotulo = sample['rotulo']

        x = torch.cat([sexo, idade, altura, peso, imc, pas, pad, glicose, creatinina, colesterol, hdl, ldl, trig], 1)


        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += rotulo.size(0)
        correct += (predicted == rotulo).sum().item()

    print('Accuracy: {} %'.format(100 * correct / total))




