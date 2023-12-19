from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling3D, LSTM, Reshape
from keras import backend as K
from keras.callbacks import History 

from keras.layers import AveragePooling2D
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import ConvLSTM2D

from matplotlib import pyplot as plt

def get_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                        input_shape=(3, 380, 346, 1),padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                    activation='linear',
                    padding='same', data_format='channels_last'))
        
    # print(model.summary())

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss=["mse"], optimizer='adam')

    return model


def fn_run_model(model, X, y, X_val, y_val, batch_size=1, nb_epoch=40,verbose=2,is_graph=True):
    history = History()
    history = model.fit(X, y, batch_size=batch_size, 
                        epochs=nb_epoch,verbose=verbose, validation_data=(X_val, y_val))
    if is_graph:
        fig, ax1 = plt.subplots(1,1)
        ax1.plot(history.history["val_loss"])
        ax1.plot(history.history["loss"])