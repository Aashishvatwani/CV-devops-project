import os
import tensorflow as tf
import tensorflow_datasets as tfds


def build_model(num_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def main() -> None:
    (train_ds, test_ds), info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    num_classes = info.features["label"].num_classes

    def prep(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.transpose(image, perm=[1, 0, 2])
        return image, label

    train_ds = (
        train_ds
        .map(prep, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10000)
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds
        .map(prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=test_ds, epochs=5)

    os.makedirs("model", exist_ok=True)
    model.save("model/letter_cnn.h5")

    with open("model/letter_labels.txt", "w", encoding="utf-8") as f:
        for i in range(1, 27):
            f.write(chr(ord("A") + i - 1) + "\n")

    print("Saved model/letter_cnn.h5 and model/letter_labels.txt")


if __name__ == "__main__":
    main()
