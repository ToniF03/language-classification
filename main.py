from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import os, pickle

encoder = LabelEncoder()

def setup_dataset():
    dataset = load_dataset("papluca/language-identification")
    train = dataset["train"]
    test = dataset["test"]
    val = dataset["validation"]

    return train, test, val

def create_model(vectorizer, verbose=0):
    vocab_size = len(vectorizer.get_vocabulary())
    model = tf.keras.Sequential([
        vectorizer,

        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=256
        ),

        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=5,
            activation="relu"
        ),

        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            len(encoder.classes_),
            activation="softmax"
        )
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
        )
    
    if verbose == 1:
        model.summary()
    
    return model

def prepare_dataset(verbose=0):
    train, test, val = setup_dataset()

    X_train = list(train["text"])
    X_test = list(test["text"])
    X_val = list(val["text"])
    y_train = encoder.fit_transform(train["labels"])
    y_val = encoder.transform(val["labels"])
    y_test = encoder.transform(test["labels"])

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="character",
        output_mode="int",
        output_sequence_length=512
    )

    if verbose == 1:
        print("Adapting Vectorizer now. This may take a while!")
    vectorizer.adapt(X_train)
    if verbose == 1:
        print("Vectorizer adapted.")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    )

    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
    )

    BATCH_SIZE = 128

    train_ds = (
        train_ds
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    for i in range(5):
        print(train["text"][i])
        print(len(train["text"][i]))
        print()

    return (train_ds, test_ds, val_ds), vectorizer
    
def train_model(verbose=0):

    (train_ds, test_ds, val_ds), vectorizer = prepare_dataset(verbose=verbose)
    model = create_model(vectorizer, verbose=verbose)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",      # Metric to monitor
        patience=3,              # Wait 3 epochs without improvement
        restore_best_weights=True,
        min_delta=0.001,         # Minimum improvement to count
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    test_ds = test_ds.batch(128)

    model.evaluate(test_ds)

    model.save("language_classifier.keras")

    return model

def predict(model, input, verbose=0):
    prediction = model.predict(tf.constant([input]), verbose=verbose)[0]

    language = encoder.inverse_transform(
        [prediction.argmax()]
    )[0]

    top = prediction.argsort()[-5:][::-1][-5:]

    print(f"Input: {input}")

    for i in top:
        print(encoder.classes_[i], prediction[i])
    print("---------------------------------------")
    return language, prediction[prediction.argmax()]

if __name__ == "__main__":
    if not os.path.isfile("label_encoder.pkl"):
        model = train_model(verbose=1)
    else:
        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        if not os.path.isfile("language_classifier.keras"):
            model = train_model(verbose=1)
        else:
            model = tf.keras.models.load_model("language_classifier.keras")

    predict(model, "This is an example")
    predict(model, "The weather is beautiful today and I would like to go for a walk.")
    predict(model, "Artificial intelligence has changed many aspects of modern computing.")
    predict(model, "I'm here the secrets that you keep")
    predict(model, "When you are talking in your sleep.")