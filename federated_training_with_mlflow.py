#!/usr/bin/env python
import os
import pickle
import tensorflow as tf
import tensorflow_federated as tff
import mlflow
import mlflow.tensorflow
from datetime import datetime

# Config
DATA_FOLDER = "federated_data"
MODEL_SAVE_PATH = "federated_data/federated_churn_model.h5"
BATCH_SIZE = 8
NUM_ROUNDS = 5
LEARNING_RATE_CLIENT = 0.01
LEARNING_RATE_SERVER = 1.0

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("federated_churn_prediction")

def create_tf_dataset(X, y, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X), y))
    return ds.batch(batch_size)

# Load datasets
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

# Get input shape
for batch in federated_train[0].take(1):
    example_input = batch[0]
    break
input_shape = example_input.shape[1]

def create_keras_model(input_shape=input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
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

# Start MLflow run
with mlflow.start_run(run_name=f"federated_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
    # Log parameters
    mlflow.log_param("num_clients", len(federated_train))
    mlflow.log_param("num_rounds", NUM_ROUNDS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("client_lr", LEARNING_RATE_CLIENT)
    mlflow.log_param("server_lr", LEARNING_RATE_SERVER)
    mlflow.log_param("model_architecture", "64-32-1")
    
    # Federated training
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CLIENT),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER)
    )
    
    state = trainer.initialize()
    
    for round_num in range(NUM_ROUNDS):
        state, metrics = trainer.next(state, federated_train)
        print(f"Round {round_num+1}, metrics: {metrics}")
        
        # Log metrics to MLflow
        mlflow.log_metric("train_loss", float(metrics['train']['loss']), step=round_num)
        mlflow.log_metric("train_accuracy", float(metrics['train']['binary_accuracy']), step=round_num)
        mlflow.log_metric("train_auc", float(metrics['train']['auc']), step=round_num)
    
    # Save model
    central_model = create_keras_model()
    
    @tf.function
    def assign_weights_to_keras_model(keras_model, global_weights):
        for var, weight in zip(keras_model.trainable_variables, global_weights.trainable):
            var.assign(weight)
        for var, weight in zip(keras_model.non_trainable_variables, global_weights.non_trainable):
            var.assign(weight)
    
    assign_weights_to_keras_model(central_model, state.global_model_weights)
    central_model.save(MODEL_SAVE_PATH)
    
    # Log model to MLflow
    mlflow.tensorflow.log_model(central_model, "model")
    mlflow.log_artifact(MODEL_SAVE_PATH)
    
    print(f"Saved federated model -> {MODEL_SAVE_PATH}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")