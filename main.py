import numpy as np

from preprocess import get_data
from model import get_model, fn_run_model


# obtain the data and model
data = get_data()
model = get_model()
print(model.summary())

# remove nan values in case any there
data[np.isnan(data)] = 0

# sample the data
num_samples = 8
num_tf = 3

sampled_data = []

for i in range(num_samples):
    sample = np.expand_dims(data[i:i+num_tf,:,:], axis = 3)
    sampled_data.append(sample)

sampled_data = np.stack(sampled_data)
print(sampled_data.shape)

train_val_split = 6

X_train_0 = sampled_data[:train_val_split]
X_train_1 = sampled_data[1:train_val_split+1]
X_val_0 = sampled_data[train_val_split:-1]
X_val_1 = sampled_data[train_val_split+1:]

fn_run_model(model, X_train_0, X_train_1, X_val_1, X_val_1, batch_size=1, nb_epoch=3,verbose=1,is_graph=True)
    

X_pred = model.predict(X_val_0)


