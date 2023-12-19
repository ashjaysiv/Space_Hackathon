from convlstm import ConvLSTM
from preprocess import get_data
import torch
import torch.nn as nn
import numpy as np
from skimage.measure import block_reduce



data = get_data()
data = np.reshape(data ,[data.shape[0], 1, 1, data.shape[1], data.shape[2]])

data = torch.tensor(data)

# print(data.shape)
input_height = data.shape[1]
input_width = data.shape[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTM(input_dim=1, hidden_dim=[64, 16] , kernel_size=(5,5), num_layers=2)

model.to(device)

model._init_hidden(batch_size=1, image_size=(input_height, input_width))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 1

train_seq_len = 7
for epoch in range(num_epochs):
    for i in range(0, train_seq_len):
        inputs = data[i:i+batch_size].to(device)
        targets = data[i+batch_size:i+2*batch_size].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        print(outputs.shape)
        # (1, 1, 32, 241, 233)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{train_seq_len}], Loss: {loss.item():.4f}')



