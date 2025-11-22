# Federated Learning Pipeline for E-commerce Customer Churn Prediction

## **Overview**
This project demonstrates a complete federated learning pipeline for predicting customer churn in an e-commerce dataset. The pipeline ensures data privacy by training models locally on client data and aggregating updates to create a global model. It is modular, scalable, and well-suited for real-world federated learning applications.

---

## **Input Data**
The input dataset is an **E-commerce Customer Churn Analysis and Prediction dataset**. It contains 20 columns and 5630 rows, with the following key features:

- **CustomerID**: Unique identifier for each customer.
- **Churn**: Target variable indicating whether a customer churned (1) or not (0).
- **Tenure**: Duration of customer relationship.
- **PreferredLoginDevice**: Device used for login (e.g., Mobile Phone, Computer).
- **CityTier**: Tier of the city where the customer resides.
- **WarehouseToHome**: Distance from the warehouse to the customer's home.
- **PreferredPaymentMode**: Payment method preferred by the customer.
- **Gender**: Gender of the customer.
- **HourSpendOnApp**: Time spent on the app.
- **NumberOfDeviceRegistered**: Number of devices registered by the customer.
- **PreferedOrderCat**: Preferred order category (e.g., Fashion, Electronics).
- **SatisfactionScore**: Customer satisfaction score.
- **Other Features**: MaritalStatus, NumberOfAddress, Complain, OrderAmountHikeFromLastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount.

The dataset contains both **numerical** and **categorical** features, with some missing values that need preprocessing.

---

## **Pipeline Overview**
The pipeline consists of three main Python scripts and a requirements file. Each script plays a specific role in the federated learning process:

---

### **1. `prepare_federated_data.py`**
This script is responsible for **data preprocessing** and **client data preparation**. It performs the following steps:

#### **Steps:**
1. **Load and Clean Data**:
   - Reads the dataset from an Excel file (`E Commerce Dataset.xlsx`).
   - Cleans and standardizes categorical values (e.g., replacing "Phone" with "Mobile Phone").
   - Drops the `CustomerID` column as it is not useful for modeling.
   - Converts the `Churn` column to integer type.

2. **Build Preprocessor**:
   - Constructs a preprocessing pipeline using `scikit-learn`:
     - **Numerical Columns**: Imputed with the median and scaled using `MinMaxScaler`.
     - **Categorical Columns**: Imputed with the most frequent value and one-hot encoded.
   - Fits the preprocessor on the entire dataset to ensure consistency across clients.

3. **Simulate Federated Clients**:
   - Shuffles the dataset and splits it into `CLIENTS` (3 in this case) subsets to simulate federated clients.
   - Each client dataset is preprocessed using the fitted preprocessor.

4. **Save Client Data**:
   - Saves each client's preprocessed data (`X`, `y`, and feature names) as `.pkl` files in the `federated_data` folder.
   - Saves the fitted preprocessor and metadata (e.g., feature names, column types) for later use.

#### **Output**:
- Preprocessed client datasets (`client_1_data.pkl`, `client_2_data.pkl`, etc.).
- Preprocessor (`preprocessor.pkl`) and metadata (`metadata.json`).

---

### **2. `federated_training.py`**
This script handles **federated learning model training** using TensorFlow Federated (TFF). It performs the following steps:

#### **Steps:**
1. **Load Client Data**:
   - Reads the preprocessed client datasets from the `federated_data` folder.
   - Converts each client's data into TensorFlow datasets (`tf.data.Dataset`), batched for training.

2. **Define Keras Model**:
   - Creates a simple feedforward neural network with the following layers:
     - Input layer (size determined by the number of features).
     - Two hidden layers with 64 and 32 neurons, respectively, using ReLU activation.
     - Output layer with a sigmoid activation for binary classification.

3. **Federated Learning Setup**:
   - Wraps the Keras model into a TFF model using `tff.learning.models.from_keras_model`.
   - Configures the **Federated Averaging (FedAvg)** algorithm:
     - **Client Optimizer**: Adam optimizer with a learning rate of 0.01.
     - **Server Optimizer**: SGD optimizer with a learning rate of 1.0.

4. **Train Federated Model**:
   - Initializes the federated learning process.
   - Trains the model for `NUM_ROUNDS` (5 in this case), where:
     - Each round aggregates model updates from all clients.
     - The server updates the global model using the aggregated updates.

5. **Save the Global Model**:
   - Extracts the global model's weights and assigns them to a Keras model.
   - Saves the trained global model as `federated_churn_model.h5`.

#### **Output**:
- Trained global model (`federated_churn_model.h5`).

---

### **3. `evaluate_federated_model.py`**
This script evaluates the **performance of the trained federated model** on a test dataset. It performs the following steps:

#### **Steps:**
1. **Load Preprocessor and Model**:
   - Loads the preprocessor (`preprocessor.pkl`) and the trained federated model (`federated_churn_model.h5`).

2. **Prepare Test Data**:
   - Reads the dataset from the Excel file and samples 20% of the data for testing.
   - Preprocesses the test data using the loaded preprocessor.

3. **Evaluate Model**:
   - Makes predictions on the test data.
   - Computes evaluation metrics:
     - **Accuracy**: Overall correctness of predictions.
     - **Precision**: Correct positive predictions out of all positive predictions.
     - **Recall**: Correct positive predictions out of all actual positives.
     - **F1-Score**: Harmonic mean of precision and recall.
     - **ROC AUC**: Area under the ROC curve.
   - Displays a classification report and confusion matrix.

4. **Visualize Results**:
   - Plots the distribution of predicted churn probabilities.
   - Computes approximate feature importance using permutation-based importance and visualizes the top features.

#### **Output**:
- Evaluation metrics (accuracy, precision, recall, F1-score, ROC AUC).
- Visualizations (churn probability distribution, feature importance).

---

### **4. `req.txt`**
This file lists the required Python packages for the project, including:
- **Core ML & Federated Learning**: `tensorflow`, `tensorflow-federated`.
- **Data Handling**: `numpy`, `pandas`, `openpyxl`.
- **Visualization**: `matplotlib`.
- **Preprocessing & Metrics**: `scikit-learn`.

---

## **Federated Learning Process**
Federated learning is a decentralized approach where the model is trained collaboratively across multiple clients without sharing their raw data. Here's how it works in this pipeline:

1. **Data Preparation**:
   - The dataset is split into multiple subsets, each representing a client.
   - Each client's data is preprocessed independently.

2. **Local Training**:
   - Each client trains the model locally on its own data.
   - The local training process uses the same model architecture and optimizer.

3. **Model Aggregation**:
   - After local training, each client sends its model updates (not raw data) to the server.
   - The server aggregates these updates using the Federated Averaging (FedAvg) algorithm to update the global model.

4. **Global Model Update**:
   - The updated global model is sent back to the clients for the next training round.
   - This process repeats for a specified number of rounds.

5. **Final Model**:
   - After training, the global model is saved and can be evaluated on a separate test dataset.

---

## **Conclusion**
This pipeline demonstrates a complete federated learning workflow for predicting customer churn in an e-commerce dataset. It ensures data privacy by keeping client data local while leveraging collaborative learning to train a robust global model. The pipeline is modular, scalable, and well-suited for real-world federated learning applications.
