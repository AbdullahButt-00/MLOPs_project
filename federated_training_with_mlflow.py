#!/usr/bin/env python
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import mlflow
import mlflow.tensorflow
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.utils import shuffle

# Optional: interactive plots
import plotly.graph_objects as go

# ===================== CONFIG =====================
DATA_FOLDER = "federated_data"
MODEL_SAVE_PATH = os.path.join(DATA_FOLDER, "federated_churn_model.h5")
BATCH_SIZE = 8
NUM_ROUNDS = 20             # change as needed
LEARNING_RATE_CLIENT = 0.01
LEARNING_RATE_SERVER = 1.0
TEST_FRAC = 0.2
RANDOM_STATE = 42

# smoothing window for moving average (odd preferred)
SMOOTH_WINDOW = 3  # set to 3; you can increase to smooth more

# ===================== MLflow =====================
# mlflow.set_tracking_uri(...) if using a remote server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("federated_churn_prediction")

# ===================== DATASET HELPERS =====================
def create_tf_dataset(X, y, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X, dtype=tf.float32),
                                             tf.convert_to_tensor(y, dtype=tf.int32)))
    return ds.batch(batch_size)

# ----------------- Load federated client datasets -----------------
client_files = sorted([
    os.path.join(DATA_FOLDER, f)
    for f in os.listdir(DATA_FOLDER)
    if f.startswith("client_") and f.endswith("_data.pkl")
])

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

# ----------------- Prepare held-out test set (fixed) -----------------
with open(os.path.join(DATA_FOLDER, "preprocessor.pkl"), "rb") as f:
    preprocessor = pickle.load(f)

raw_df = pd.read_excel("E Commerce Dataset.xlsx", sheet_name="E Comm")
raw_df['Churn'] = raw_df['Churn'].astype(int)
raw_df = shuffle(raw_df, random_state=RANDOM_STATE)

test_df = raw_df.sample(frac=TEST_FRAC, random_state=RANDOM_STATE)
X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn'].values
X_test_trans = preprocessor.transform(X_test).astype(np.float32)
print(f"Prepared test split: {len(y_test)} rows")

# ----------------- Determine input shape -----------------
for batch in federated_train[0].take(1):
    example_input = batch[0]
    break
input_shape = int(example_input.shape[1])

# ===================== MODEL DEFINITION =====================
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

# ===================== TRAINING & MLflow RUN =====================
round_eval_dir = os.path.join(DATA_FOLDER, "round_evaluation")
os.makedirs(round_eval_dir, exist_ok=True)

with mlflow.start_run(run_name=f"federated_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

    # Log params
    mlflow.log_param("num_clients", len(federated_train))
    mlflow.log_param("num_rounds", NUM_ROUNDS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("client_lr", LEARNING_RATE_CLIENT)
    mlflow.log_param("server_lr", LEARNING_RATE_SERVER)
    mlflow.log_param("test_frac", TEST_FRAC)
    mlflow.log_param("model_architecture", "64-32-1")
    mlflow.log_param("smoothing_window", SMOOTH_WINDOW)

    # Prepare trainer
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CLIENT),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER)
    )

    state = trainer.initialize()

    # containers for per-round evaluation
    rounds = []
    train_losses = []
    train_accuracies = []
    train_aucs = []

    eval_accuracies = []
    eval_precisions = []
    eval_recalls = []
    eval_f1s = []
    eval_roc_aucs = []

    for round_num in range(NUM_ROUNDS):
        # perform federated round
        state, metrics = trainer.next(state, federated_train)
        print(f"Round {round_num+1}, metrics: {metrics}")

        # log training-side metrics (from clients aggregate)
        train_metrics = metrics['client_work']['train']
        train_loss = float(train_metrics['loss'])
        train_acc = float(train_metrics['binary_accuracy'])
        train_auc = float(train_metrics['auc'])

        mlflow.log_metric("train_loss", train_loss, step=round_num)
        mlflow.log_metric("train_accuracy", train_acc, step=round_num)
        mlflow.log_metric("train_auc", train_auc, step=round_num)

        # keep in lists for plotting
        rounds.append(round_num + 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_aucs.append(train_auc)

        # ---------------- Evaluate the GLOBAL model (current state) ----------------
        # Build a fresh Keras model and assign global weights
        central_model = create_keras_model()
        global_weights = state.global_model_weights

        # assign trainable and non-trainable
        for var, val in zip(central_model.trainable_variables, global_weights.trainable):
            var.assign(val)
        for var, val in zip(central_model.non_trainable_variables, global_weights.non_trainable):
            var.assign(val)

        # Predict on the held-out test set
        y_pred_prob = central_model.predict(X_test_trans, batch_size=BATCH_SIZE).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        # Log evaluation metrics per round to MLflow (use step as round_num)
        mlflow.log_metric("eval_accuracy", acc, step=round_num)
        mlflow.log_metric("eval_precision", prec, step=round_num)
        mlflow.log_metric("eval_recall", rec, step=round_num)
        mlflow.log_metric("eval_f1", f1, step=round_num)
        mlflow.log_metric("eval_roc_auc", roc_auc, step=round_num)

        # keep for plotting
        eval_accuracies.append(acc)
        eval_precisions.append(prec)
        eval_recalls.append(rec)
        eval_f1s.append(f1)
        eval_roc_aucs.append(roc_auc)

        # Save per-round metrics row to CSV (append)
        row = {
            "round": round_num + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "eval_accuracy": acc,
            "eval_precision": prec,
            "eval_recall": rec,
            "eval_f1": f1,
            "eval_roc_auc": roc_auc
        }
        df_row = pd.DataFrame([row])
        metrics_csv = os.path.join(round_eval_dir, "per_round_metrics.csv")
        if round_num == 0:
            df_row.to_csv(metrics_csv, index=False)
        else:
            df_row.to_csv(metrics_csv, mode='a', header=False, index=False)

    # ===================== After all rounds: Save final global model =====================
    central_model.save(MODEL_SAVE_PATH)
    print(f"Saved federated model -> {MODEL_SAVE_PATH}")
    mlflow.tensorflow.log_model(central_model, artifact_path="model")
    mlflow.log_artifact(MODEL_SAVE_PATH)

    # ----------------- Create aesthetic matplotlib plots with smoothing + CI -----------------
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-darkgrid")

    def smooth_and_ci(series, window=SMOOTH_WINDOW):
        s = pd.Series(series)
        # moving average (centered)
        smooth = s.rolling(window=window, center=True, min_periods=1).mean()
        # rolling std
        std = s.rolling(window=window, center=True, min_periods=1).std().fillna(0.0)
        # 95% approx band using 1.96 * std (note: not true CI across runs, but indicates variability)
        upper = smooth + 1.96 * std
        lower = smooth - 1.96 * std
        return smooth.values, lower.values, upper.values

    def plot_with_ci(x, y, ylabel, title, outpath_png, outpath_svg):
        smooth, lower, upper = smooth_and_ci(y, window=SMOOTH_WINDOW)
        plt.figure(figsize=(9,5))
        plt.plot(x, y, marker='o', linestyle='-', alpha=0.35, label='raw', linewidth=1.5)
        plt.plot(x, smooth, marker='o', linestyle='-', linewidth=2.2, label='smoothed')
        plt.fill_between(x, lower, upper, color='gray', alpha=0.25, label='±1.96·std (rolling)')
        plt.xticks(x)
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath_png, dpi=160)
        plt.savefig(outpath_svg)
        plt.close()

    # craft metric plots
    plot_with_ci(rounds, eval_accuracies, "Accuracy", "Eval Accuracy vs Round (smoothed)", os.path.join(round_eval_dir, "eval_accuracy_vs_round.png"), os.path.join(round_eval_dir, "eval_accuracy_vs_round.svg"))
    plot_with_ci(rounds, eval_precisions, "Precision", "Eval Precision vs Round (smoothed)", os.path.join(round_eval_dir, "eval_precision_vs_round.png"), os.path.join(round_eval_dir, "eval_precision_vs_round.svg"))
    plot_with_ci(rounds, eval_recalls, "Recall", "Eval Recall vs Round (smoothed)", os.path.join(round_eval_dir, "eval_recall_vs_round.png"), os.path.join(round_eval_dir, "eval_recall_vs_round.svg"))
    plot_with_ci(rounds, eval_f1s, "F1-score", "Eval F1-score vs Round (smoothed)", os.path.join(round_eval_dir, "eval_f1_vs_round.png"), os.path.join(round_eval_dir, "eval_f1_vs_round.svg"))
    plot_with_ci(rounds, eval_roc_aucs, "ROC AUC", "Eval ROC AUC vs Round (smoothed)", os.path.join(round_eval_dir, "eval_roc_auc_vs_round.png"), os.path.join(round_eval_dir, "eval_roc_auc_vs_round.svg"))

    # ----------------- Create interactive Plotly chart with raw + smoothed + CI -----------------
    def make_plotly_combined(rounds, metric_dict, out_html):
        """
        metric_dict: dict of name -> list(values)
        Creates an interactive chart with traces for raw and smoothed series plus shaded CI.
        """
        fig = go.Figure()
        for name, values in metric_dict.items():
            s = pd.Series(values)
            smooth = s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
            std = s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).std().fillna(0.0)
            upper = (smooth + 1.96 * std).values
            lower = (smooth - 1.96 * std).values

            # raw trace
            fig.add_trace(go.Scatter(x=rounds, y=values, mode='markers+lines', name=f"{name} (raw)", opacity=0.4))
            # smoothed trace
            fig.add_trace(go.Scatter(x=rounds, y=smooth, mode='lines+markers', name=f"{name} (smoothed)", line=dict(width=3)))
            # CI band (fill between lower and upper)
            fig.add_trace(go.Scatter(
                x=rounds + rounds[::-1],
                y=list(upper) + list(lower[::-1]),
                fill='toself',
                fillcolor='rgba(150,150,150,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} CI"
            ))

        fig.update_layout(
            title="Per-round Evaluation Metrics (raw + smoothed + CI)",
            xaxis_title="Round",
            yaxis_title="Metric value",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.write_html(out_html, include_plotlyjs='cdn')

    metrics_for_plotly = {
        "Accuracy": eval_accuracies,
        "Precision": eval_precisions,
        "Recall": eval_recalls,
        "F1": eval_f1s,
        "ROC AUC": eval_roc_aucs
    }
    plotly_html = os.path.join(round_eval_dir, "per_round_metrics_interactive.html")
    make_plotly_combined(rounds, metrics_for_plotly, plotly_html)

    # ---------------- Save and log artifacts to MLflow -----------------
    # Save per-round CSV has been written during loop
    mlflow.log_artifact(metrics_csv, artifact_path="round_evaluation")

    # Log all plot files in round_eval_dir
    for fn in os.listdir(round_eval_dir):
        mlflow.log_artifact(os.path.join(round_eval_dir, fn), artifact_path="round_evaluation")

    print("\nPer-round evaluation metrics, plots (matplotlib & plotly) saved to:", round_eval_dir)
    print("They have also been logged to MLflow under artifact path 'round_evaluation'.")

    # Also log final eval metrics under final_*
    if len(eval_accuracies) > 0:
        mlflow.log_metric("final_accuracy", eval_accuracies[-1])
        mlflow.log_metric("final_precision", eval_precisions[-1])
        mlflow.log_metric("final_recall", eval_recalls[-1])
        mlflow.log_metric("final_f1", eval_f1s[-1])
        mlflow.log_metric("final_roc_auc", eval_roc_aucs[-1])

    # ===================== END RUN =====================
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
