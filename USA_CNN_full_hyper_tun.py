import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad, Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

# Generate Fibonacci sequence up to a certain length
def generate_fibonacci(length):
    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < length:
        fibonacci_sequence.append(fibonacci_sequence[-1] + fibonacci_sequence[-2])
    return fibonacci_sequence[:length]

# Load the data
data = pd.read_csv('')

# Ensure date is datetime type
data['date'] = pd.to_datetime(data['date'])

# Set the date as the index
data.set_index('date', inplace=True)

# Preprocess the data
features = data.values

# Scale the original features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Concatenate Fibonacci sequence with existing features
fibonacci_sequence = generate_fibonacci(7)
features_with_fibonacci = np.concatenate((features_scaled, np.array(fibonacci_sequence).reshape(1, -1).repeat(len(features_scaled), axis=0)), axis=1)

# Function to create sequences for time series data
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTHS = [5, 8, 13, 21]
EPOCHS_LIST = [144, 233, 377]
BATCH_SIZES = [34, 55, 89]
OPTIMIZERS = ['adam', 'sgd', 'adadelta', 'adagrad', 'nadam']

optimizer_options = {
    'adam': Adam,
    'sgd': SGD,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'nadam': Nadam
}

best_params = {'seq_length': None, 'epochs': None, 'batch_size': None, 'optimizer': None}
best_score = float('inf')

X = features_with_fibonacci
y = features_with_fibonacci

for seq_length in SEQ_LENGTHS:
    X_seq, y_seq = create_sequences(features_with_fibonacci, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    def build_model(seq_length, optimizer_name):
        optimizer_class = optimizer_options[optimizer_name]
        optimizer = optimizer_class()  # Create optimizer instance here
        model = Sequential([
            Input(shape=(seq_length, X_train.shape[2])),
            Conv1D(filters=64, kernel_size=2, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(X_train.shape[2])
        ])
        model.compile(optimizer=optimizer, loss='mse')
        return model

    for epochs in EPOCHS_LIST:
        for batch_size in BATCH_SIZES:
            for optimizer_name in OPTIMIZERS:
                print(f"Testing seq_length={seq_length}, epochs={epochs}, batch_size={batch_size}, optimizer={optimizer_name}")

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = []

                for train_idx, val_idx in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                    model = build_model(seq_length, optimizer_name)
                    history = model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)

                    y_pred_val = model.predict(X_val_fold)
                    val_loss = mean_squared_error(y_val_fold, y_pred_val)
                    scores.append(val_loss)

                avg_score = np.mean(scores)
                print(f"Average Validation Loss: {avg_score}")

                if avg_score < best_score:
                    best_score = avg_score
                    best_params['seq_length'] = seq_length
                    best_params['epochs'] = epochs
                    best_params['batch_size'] = batch_size
                    best_params['optimizer'] = optimizer_name

print(f"Best Parameters: {best_params}")
print(f"Best Validation Loss: {best_score}")

best_seq_length = best_params['seq_length']
best_epochs = best_params['epochs']
best_batch_size = best_params['batch_size']
best_optimizer_name = best_params['optimizer']
best_optimizer = optimizer_options[best_optimizer_name]()

X_seq, y_seq = create_sequences(features_with_fibonacci, best_seq_length)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

final_model = build_model(best_seq_length, best_optimizer_name)
final_history = final_model.fit(X_train, y_train, epochs=best_epochs, batch_size=best_batch_size, verbose=1, validation_split=0.2)


# Evaluate the final model on the test set
final_loss = final_model.evaluate(X_test, y_test)
print(f"Final Test Loss: {final_loss}")

# Make predictions on the test set
predictions = final_model.predict(X_test)

# Inverse transform the predictions and actual values to compare
# Separate the original scaled features and Fibonacci sequence before inverse transform
X_test_orig = X_test[:, :, :features_scaled.shape[1]]  # Original scaled features
X_test_fibonacci = X_test[:, :, features_scaled.shape[1]:]  # Fibonacci sequence

# Inverse transform the original scaled features
y_test_inv_orig = scaler.inverse_transform(y_test[:, :features_scaled.shape[1]])

# Inverse transform the predictions
predictions_inv_orig = scaler.inverse_transform(predictions[:, :features_scaled.shape[1]])

# Convert predictions to integers
predictions_inv_int_orig = predictions_inv_orig.astype(int)

# Apply constraints
predictions_inv_int_orig[:, :5] = np.clip(predictions_inv_int_orig[:, :5], 1, 59)  # For num1 to num5, clip values to [1, 50]
predictions_inv_int_orig[:, 5:] = np.clip(predictions_inv_int_orig[:, 5:], 1, 35)  # For num6 to num7, clip values to [1, 12]

predi = pd.DataFrame(predictions_inv_int_orig)
today = datetime.today()
predi.to_csv(f'')
# Display the results
print("Predicted values (int with constraints):\n", predictions_inv_int_orig)

# Predict future values for a specific date
def predict_for_date(start_sequence, model, days_ahead):
    current_sequence = start_sequence.copy()
    predictions = []

    for _ in range(days_ahead):
        prediction = model.predict(current_sequence.reshape(1, best_seq_length, start_sequence.shape[1]))

        prediction_inv = scaler.inverse_transform(prediction[:, :features_scaled.shape[1]])
        prediction_int = prediction_inv.astype(int)

        # Remove duplicates in the prediction
        unique_prediction = []
        for value in prediction_int[0]:
            while value in unique_prediction:
                value += 1  # Adjust value to remove duplicate
            unique_prediction.append(value)
        prediction_int[0] = np.array(unique_prediction)

        prediction_int[:, :5] = np.clip(prediction_int[:, :5], 1, 59)
        prediction_int[:, 5:] = np.clip(prediction_int[:, 5:], 1, 35)

        predictions.append(prediction_int[0])

        new_entry = np.concatenate((prediction[0], np.array(fibonacci_sequence)), axis=0)  # Corrected this line
        new_entry = new_entry[:start_sequence.shape[1]]  # Ensure new_entry matches the shape

        current_sequence = np.vstack([current_sequence[1:], new_entry])

    return predictions


print("Today's date:", today)

days_ahead = 7

last_sequence = features_with_fibonacci[-best_seq_length:]

future_predictions = predict_for_date(last_sequence, final_model, days_ahead)

future_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]
for date, prediction in zip(future_dates, future_predictions):
    print(f"Prediction for {date.strftime('%Y-%m-%d')}: {prediction}")

# Create a DataFrame
predictions_df = pd.DataFrame(prediction)

# Save the DataFrame to a CSV file
predictions_df.to_csv(f'', index=False)

print(f'Predictions saved to ')
