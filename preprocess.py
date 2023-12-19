"""
    Preprocess the data for the model
"""

import rasterio
import numpy as np
from scipy.ndimage import maximum_filter
from skimage.measure import block_reduce
import os
from matplotlib import pyplot as plt

def get_mask():
    """
        Take the file with the minimum nan values and utilize mak to ensure uniformity: "1415_vik.tif"
    """

    file_path = "./LULC_2005_15_vik/1415_vik.tif"
    
    with rasterio.open(file_path) as src:
        # Read the grayscale band
        data = src.read(1) 

    reduced_data = block_reduce(data, block_size = (3,3), func = lambda block, axis: maxfreq_pool(block, axis))

    # mark boundary
    for i in range(len(reduced_data)):
        for j in range(len(reduced_data[0])):
            if reduced_data[i][j] == 0:
                reduced_data[i][j] = np.nan

    # create mask
    mask = ~np.isnan(reduced_data)

    return mask




def maxfreq_pool(block, axis = None):
    length = block.shape[0]
    width = block.shape[1]

    values = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
                element = block[i][j]
                unique_values, counts = np.unique(element, return_counts=True)
                index_of_max_frequency = np.argmax(counts)
                values[i][j] = unique_values[index_of_max_frequency]

    return values

def process(file_path):
    
    # Open the GeoTIFF file

    with rasterio.open(file_path) as src:
        # Read the grayscale band
        data = src.read(1)  # Change '1' to the band number you want to use

    reduced_data = block_reduce(data, block_size = (3,3), func = lambda block, axis: maxfreq_pool(block, axis))

    # mark boundary
    for i in range(len(reduced_data)):
        for j in range(len(reduced_data[0])):
            if reduced_data[i][j] == 0:
                reduced_data[i][j] = np.nan

    # create mask
    mask = get_mask()

    # ignoring np.nan values
    masked_data = reduced_data[mask]

    # Reshape the new array to a 2D shape
    flattened_data = masked_data.reshape(np.sum(mask), -1)
    padding = 380*346 - flattened_data.shape[0]
    zeros = np.zeros( padding )
    zeros = zeros.reshape((padding,1))
    final_data = np.concatenate((zeros, flattened_data), axis = 0)
    final_data = final_data.reshape((380,346))
    final_data = final_data/ 17 # normalize
    
    
    # print(final.shape) # 362*363
    # print(reduced_data.shape) # 483*487
    # print(data.shape)  # 1499*1399
    
    # # to visualize the data
    # plt.imshow(reduced_data, cmap="viridis")
    # plt.colorbar()
    # plt.show()      

    # 346*380

    return final_data

def get_data():

    dirs = sorted(os.listdir("./LULC_2005_15_vik"))
    stacked_data = []
    for i, file in enumerate(dirs):
        path = "./LULC_2005_15_vik/" + file
        data = process(path)
        # cropped_data = data[:len(data)//2,:len(data[0])//2]
        stacked_data.append(data)

    return np.stack(stacked_data)

