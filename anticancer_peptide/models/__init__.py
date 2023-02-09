from tensorflow import keras

def create_model(vocab_size, embedding_dim, input_length):

    model = keras.models.Sequential()
    model.add(
        keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=input_length)
    )
    model.add(
        keras.layers.Conv1D(512, 8)
    )
    model.add(keras.layers.Dropout(0.5))
    model.add(
        keras.layers.Conv1D(256, 8)
    )
    model.add(keras.layers.Dropout(0.5))
    model.add(
        keras.layers.Conv1D(128, 8)
    )
    model.add(keras.layers.Dropout(0.5))
    model.add(
        keras.layers.Conv1D(64, 8)
    )
    model.add(keras.layers.Dropout(0.5))
    model.add(
        keras.layers.Conv1D(32, 4)
    )
    model.add(keras.layers.Dropout(0.5))
    model.add(
        keras.layers.Flatten()
    )
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model