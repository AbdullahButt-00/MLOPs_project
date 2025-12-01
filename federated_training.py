# update this file:
# federated_training.py:

#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

# ------------------- Config -------------------
DATA_FOLDER = "federated_data"
MODEL_SAVE_PATH = "federated_data/federated_churn_model.h5"
BATCH_SIZE = 8
NUM_ROUNDS = 5
RANDOM_STATE = 42

def create_tf_dataset(X, y, batch_size=BATCH_SIZE, shuffle_buffer=100):
    # Ensure correct dtypes for TFF
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    # per-client shuffle/repeat can be helpful for small datasets:
    # ds = ds.shuffle(shuffle_buffer, seed=RANDOM_STATE).repeat(1)
    return ds.batch(batch_size)

# ------------------- Load datasets -------------------
client_files = sorted([os.path.join(DATA_FOLDER, f) 
                for f in os.listdir(DATA_FOLDER) 
                if f.startswith("client_") and f.endswith("_data.pkl")])

if not client_files:
    raise RuntimeError(f"No client files found in {DATA_FOLDER}. Run prepare_federated_data.py first.")

federated_train = []

for file in client_files:
    with open(file, "rb") as f:
        data = pickle.load(f)
        X, y = data['X'], data['y']
        ds = create_tf_dataset(X, y)
        federated_train.append(ds)

print(f"Loaded {len(federated_train)} federated client datasets")

# ------------------- Define Keras model -------------------
# Get input shape from first element_spec
example_spec = federated_train[0].element_spec
# element_spec is a tuple (features_tensor_spec, label_tensor_spec)
feature_spec = example_spec[0]
# feature_spec.shape is (None, num_features)
num_features = int(feature_spec.shape[-1])

def create_keras_model(input_shape=num_features):
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
global_weights = state.global_model_weights

# assign trainable and non-trainable variables
for var, val in zip(central_model.trainable_variables, global_weights.trainable):
    var.assign(val)
for var, val in zip(central_model.non_trainable_variables, global_weights.non_trainable):
    var.assign(val)

central_model.save(MODEL_SAVE_PATH)
print(f"Saved federated model -> {MODEL_SAVE_PATH}")



