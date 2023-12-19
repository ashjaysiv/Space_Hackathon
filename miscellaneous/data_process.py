"""
    To make data as a csv file
"""

import rasterio
import numpy as np
from scipy.ndimage import maximum_filter
from skimage.measure import block_reduce
import os
import csv

def process(file_path):
    
    # Open the GeoTIFF file

    with rasterio.open(file_path) as src:
        # Read the grayscale band
        data = src.read(1)  # Change '1' to the band number you want to use

    # Define 18 equally spaced boundaries for categorization
    boundaries = np.linspace(data.min(), data.max(), 18)

    # Assign labels based on the boundaries
    labels = np.digitize(data, bins=boundaries)

    labels = maximum_filter(labels, size=3)

    labels = block_reduce(labels, block_size=(3, 3), func=np.max)
    print(type(labels))

    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i][j] == 1:
                labels[i][j] = 0

            else:
                break

    for i in range(len(labels)):
        for j in range(len(labels[0])-1, -1, -1):
            if labels[i][j] == 1:
                labels[i][j] = 0

            else:
                break
                
    return labels

def main():
    path = "./LULC_2005_15_vik/0506_vik.tif"
    pixel_wise_class = process(path)

    with open('train_data_0.csv', 'w',) as output:
        writer = csv.writer(output, lineterminator='\n')

        for idx_1, row in enumerate(pixel_wise_class):
            for idx_2, element in enumerate(row):
                data = idx_1*len(row) + idx_2 

                if element != 0:                  
                    writer.writerow([data])

    print(sorted(os.listdir("./LULC_2005_15_vik")))
          
    dirs = sorted(os.listdir("./LULC_2005_15_vik"))[-3:]

    for i, file in enumerate(dirs):
        print(file)
        with open (f'train_data_{i}.csv', 'r') as input_:
            reader_ = csv.reader(input_)

            with open(f'train_data_{i+1}.csv', 'w') as output:
                writer = csv.writer(output, lineterminator='\n')


                path = "./LULC_2005_15_vik/" + file
                pixel_wise_class = process(path)

                for idx_1, row in enumerate(pixel_wise_class):
                    for idx_2, element in enumerate(row):
                        if element != 0:                  
                            # writer.writerow([str(data), element])
                            row_ = next(reader_)
                            

                            row_.append(str(element))
                            writer.writerow(row_)
                             
                os.remove(f"train_data_{i}.csv")



                
                            
   
        

if __name__ == "__main__":
    main()

# # Path to your grayscale GeoTIFF file
# file_path = '/home/ashmitha/Downloads/topic13/LULC_2005_15_vik/0506_vik.tif'


# class Archi_3GRU16BI_1FC256(nn.Module):
#     def __init__(self, input_size, nbclasses, nbunits_rnn=16, nbunits_fc=256, nb_rnn=3, dropout_rate=0.5, l2_rate=1e-6):
#         super(Archi_3GRU16BI_1FC256, self).__init__()
        
#         # Parameters
#         self.nbunits_rnn = nbunits_rnn
#         self.nbunits_fc = nbunits_fc
#         self.nb_rnn = nb_rnn
#         self.dropout_rate = dropout_rate
#         self.l2_rate = l2_rate
        
#         # GRU layers
#         self.gru_layers = nn.ModuleList()
#         for i in range(nb_rnn):
#             input_size = input_size if i == 0 else nbunits_rnn * 2  # Doubles input size for bidirectional GRU
#             self.gru_layers.append(nn.GRU(input_size, nbunits_rnn, bidirectional=True if i < nb_rnn - 1 else False))
#             self.gru_layers.append(nn.Dropout(dropout_rate))
        
#         # Fully connected layer
#         self.fc = nn.Linear(nbunits_rnn * 2, nbunits_fc)
#         self.fc_bn = nn.BatchNorm1d(nbunits_fc)
#         self.fc_relu = nn.ReLU()
#         self.fc_dropout = nn.Dropout(dropout_rate)
        
#         # Softmax output layer
#         self.softmax = nn.Linear(nbunits_fc, nbclasses)
        
#     def forward(self, x):
#         # GRU layers
#         for i, layer in enumerate(self.gru_layers):
#             if i % 2 == 0:  # GRU layers
#                 x, _ = layer(x)
#             else:  # Dropout layers
#                 x = layer(x)
        
#         # Flatten the output for the FC layer
#         x = x.view(x.size(0), -1)
        
#         # Fully connected layer
#         x = self.fc(x)
#         x = self.fc_bn(x)
#         x = self.fc_relu(x)
#         x = self.fc_dropout(x)
        
#         # Softmax output
#         x = self.softmax(x)
#         return x
