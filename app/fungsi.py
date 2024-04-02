import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Activation,
    Dropout,
)


def make_model():
    # from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications import InceptionResNetV2

    # Memuat model InceptionResNetV2 yang telah dilatih sebelumnya
    # inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    inceptionresnetv2 = InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # Mematikan pembelajaran pada layer-layer yang telah dilatih sebelumnya
    inceptionresnetv2.trainable = False
    # inceptionv3.trainable = False

    # Membuat model baru
    model = tf.keras.Sequential(
        [
            # inceptionv3,
            inceptionresnetv2,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def make_model2():
    from tensorflow.keras.applications import InceptionV3

    # Memuat model InceptionResNetV2 yang telah dilatih sebelumnya
    inceptionv3 = InceptionV3(
        weights="imagenet", include_top=False, input_shape=(300, 300, 3)
    )

    # Mematikan pembelajaran pada layer-layer yang telah dilatih sebelumnya
    inceptionv3.trainable = False

    # Membuat model baru
    model = tf.keras.Sequential(
        [
            inceptionv3,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(0, 5),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0, 5),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0, 5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0, 5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model
