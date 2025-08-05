import streamlit as st
import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from io import StringIO

# Streamlit app configuration
st.set_page_config(page_title="DT Log Prediction", layout="wide")
st.title("Sonic Log (DT) Prediction using Neural Networks")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload LAS File", type=['las'])
    
    # Model parameters
    st.subheader("Model Parameters")
    epochs = st.slider("Number of Epochs", 50, 500, 100)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    test_size = st.slider("Test Size Ratio", 0.1, 0.3, 0.2)
    random_state = st.number_input("Random State", 42)

# Main content area
if uploaded_file is not None:
    try:
        # Read LAS file
        las_text = uploaded_file.read().decode('utf-8')
        las = lasio.read(StringIO(las_text))
        
        # Extract logs
        depth = las['DEPTH']
        gr = las['GR']
        rt = las['RT']
        phie = las['PHIE']
        nphi = las['NPHI']
        vcl = las['VCL']
        rhob = las['RHOBMOD']
        sw = las['SW']
        dt_real = las['DT']
        dtsm = las['DTSM']

        # Prepare data
        X = np.vstack([gr, rt, phie, nphi, vcl, rhob, sw, dtsm]).T
        y = dt_real.reshape(-1, 1)

        # Handle missing values
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        y = np.nan_to_num(y, nan=np.nanmean(y))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Normalize
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)

        # Build and train model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train with progress bar
        st.subheader("Model Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: (
                    progress_bar.progress((epoch + 1) / epochs),
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")
                )
            )]
        )

        # Predictions
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = scaler_y.inverse_transform(y_test)

        # Metrics
        r2 = r2_score(y_test_original, y_pred)
        mae = np.mean(np.abs(y_test_original - y_pred))
        rmse = np.sqrt(np.mean((y_test_original - y_pred)**2))

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Performance")
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("MAE (μs/ft)", f"{mae:.4f}")
            st.metric("RMSE (μs/ft)", f"{rmse:.4f}")

        # Create DataFrame with predictions
        test_indices = np.arange(len(X))[train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=random_state
        )[1]]
        
        results_df = pd.DataFrame({
            'DEPTH': depth[test_indices],
            'DT_REAL': y_test_original.flatten(),
            'DT_PRED': y_pred.flatten(),
            'ERROR': (y_pred - y_test_original).flatten()
        })

        # Download button
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='dt_predictions.csv',
            mime='text/csv'
        )

        # Plots
        st.subheader("Results Visualization")
        
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        sc = ax1.scatter(y_test_original, y_pred, c=scaler_X.inverse_transform(X_test)[:, 6], cmap='jet_r')
        plt.colorbar(sc, label='SW')
        ax1.plot([min(y_test_original), max(y_test_original)], 
                [min(y_test_original), max(y_test_original)], 'r--', label='1:1 Line')
        ax1.set_xlabel('Real DT (μs/ft)')
        ax1.set_ylabel('Predicted DT (μs/ft)')
        ax1.set_title(f'Real vs Predicted DT (R² = {r2:.4f})')
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(5, 10))
        ax2.plot(y_test_original, results_df['DEPTH'], 'b-', label='Real DT')
        ax2.plot(y_pred, results_df['DEPTH'], 'r--', label='Predicted DT')
        ax2.invert_yaxis()
        ax2.set_xlabel('DT (μs/ft)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Real and Predicted DT Log')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a LAS file to get started")

# Add some app information
st.sidebar.markdown("""
**App Information:**
- Predicts DT log using GR, RT, PHIE, NPHI, VCL, RHOBMOD, SW, DTSM
- Uses a 3-layer neural network with dropout
- Outputs predictions as downloadable CSV
""")
