import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .layers import TransformerBlock, TokenAndPositionEmbedding

def build_model(output_shape, vocab_size = 24, maxlen = 60, embed_dim = 16, num_heads = 4, ff_dim = 32):

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

    x = embedding_layer(inputs)
    transformer_block_1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    transformer_block_2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    transformer_block_3 = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block_1(x)
    x = layers.Dropout(0.1)(x)
    x = transformer_block_2(x)
    x = layers.Dropout(0.1)(x)
    x = transformer_block_3(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(output_shape, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model