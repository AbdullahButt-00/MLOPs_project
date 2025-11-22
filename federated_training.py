#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import tensorflow as tf
import tensorflow_federated as tff

# ------------------- Config -------------------
DATA_FOLDER = "federated_data"
MODEL_SAVE_PATH = "federated_data/federated_churn_model.h5"
BATCH_SIZE = 8
NUM_ROUNDS = 5

def create_tf_dataset(X, y, batch_size=BATCH_SIZE):
    # Convert X to a TensorFlow tensor directly
    ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X), y))
    return ds.batch(batch_size)

# ------------------- Load datasets -------------------
client_files = [os.path.join(DATA_FOLDER, f) 
                for f in os.listdir(DATA_FOLDER) 
                if f.startswith("client_") and f.endswith("_data.pkl")]
federated_train = []

for file in client_files:
    with open(file, "rb") as f:
        data = pickle.load(f)
        X, y = data['X'], data['y']
        ds = create_tf_dataset(X, y)
        federated_train.append(ds)

print(f"Loaded {len(federated_train)} federated client datasets")

# ------------------- Define Keras model -------------------
# Get input shape from first batch
for batch in federated_train[0].take(1):
    example_input = batch[0]
    break
input_shape = example_input.shape[1]  # Use the second dimension for the number of features

def create_keras_model(input_shape=input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=federated_train[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )

# ------------------- Federated training -------------------
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

state = trainer.initialize()

for round_num in range(NUM_ROUNDS):
    state, metrics = trainer.next(state, federated_train)
    print(f"Round {round_num+1}, metrics: {metrics}")

# ------------------- Save central model -------------------
central_model = create_keras_model()

# Assign weights from the global model to the Keras model
@tf.function
def assign_weights_to_keras_model(keras_model, global_weights):
    for var, weight in zip(keras_model.trainable_variables, global_weights.trainable):
        var.assign(weight)
    for var, weight in zip(keras_model.non_trainable_variables, global_weights.non_trainable):
        var.assign(weight)

assign_weights_to_keras_model(central_model, state.global_model_weights)

# Save the Keras model
central_model.save(MODEL_SAVE_PATH)
print(f"Saved federated model -> {MODEL_SAVE_PATH}")



