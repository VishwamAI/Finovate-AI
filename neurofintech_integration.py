import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from NeuroFlex.advanced_thinking import NeuroFlexNN, create_train_state
from sklearn.preprocessing import StandardScaler

class NeuroFintech:
    def __init__(self, input_data, output_dim):
        self.scaler = StandardScaler()
        self.preprocessed_data = self.preprocess_data(input_data)
        input_dim = self.preprocessed_data.shape[1]
        self.model = NeuroFlexNN(features=[input_dim, 64, 32, output_dim], use_cnn=False)
        self.rng = random.PRNGKey(0)
        dummy_input = jnp.ones((1, input_dim))
        self.state = create_train_state(self.rng, self.model, dummy_input, learning_rate=1e-3)
        self.params = self.state.params
        self.opt_state = self.state.opt_state
        print(f"Model initialized with input dimension: {input_dim}")

    def preprocess_data(self, data):
        # Convert timestamp to numeric (Unix timestamp)
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp']).astype(int) / 10**9

        # Normalize numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        # One-hot encode categorical features
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        data = pd.get_dummies(data, columns=categorical_cols)

        # Convert all columns to float32
        for col in data.columns:
            data[col] = data[col].astype(np.float32)

        # Ensure all data is numeric
        non_numeric = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Non-numeric data found in columns: {non_numeric}")

        return jnp.array(data.values, dtype=jnp.float32)

    def train(self, X, y, epochs=100, batch_size=32):
        @jax.jit
        def loss_fn(params, X_batch, y_batch):
            logits = self.model.apply({'params': params}, X_batch)
            return jnp.mean((logits - y_batch) ** 2)

        @jax.jit
        def train_step(state, X_batch, y_batch):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, X_batch, y_batch)
            state = state.apply_gradients(grads=grads)
            return state, loss

        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.state, loss = train_step(self.state, X_batch, y_batch)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        X_preprocessed = self.preprocess_data(X)
        return self.model.apply({'params': self.params}, X_preprocessed)

def analyze_transactions(transactions_df):
    # Preprocess data
    X = transactions_df.drop('amount', axis=1)
    y = transactions_df['amount'].values

    # Initialize NeuroFintech with preprocessed input data
    neurofintech = NeuroFintech(input_data=X, output_dim=1)

    # Get preprocessed data for training
    preprocessed_X = neurofintech.preprocess_data(X)
    print(f"Preprocessed input shape: {preprocessed_X.shape}")

    # Convert to jax arrays
    X = jnp.array(preprocessed_X)
    y = jnp.array(y)

    # Train the model
    neurofintech.train(X, y)

    # Make predictions
    predictions = neurofintech.predict(X)

    # Analyze results
    mse = jnp.mean((predictions - y) ** 2)
    print(f"Mean Squared Error: {mse}")

    return predictions, mse

if __name__ == "__main__":
    # Example usage
    transactions_df = pd.DataFrame({
        'user_id': range(1000),
        'transaction_type': np.random.choice(['deposit', 'withdrawal', 'transfer'], 1000),
        'amount': np.random.uniform(10, 1000, 1000),
        'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H')
    })

    predictions, mse = analyze_transactions(transactions_df)
    print("Transaction analysis completed.")
