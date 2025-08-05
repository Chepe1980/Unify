import streamlit as st
import lasio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from io import StringIO, BytesIO

# Configure the app
st.set_page_config(
    page_title="DT Prediction with 1D CNN",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1rem;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .download-buttons {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("⛏️ Sonic Log (DT) Prediction using 1D CNN")
    st.markdown("""
    This app predicts sonic travel time (DT) from other well log measurements using a 1D Convolutional Neural Network.
    Upload your LAS file and configure the model parameters in the sidebar.
    """)
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'original_las' not in st.session_state:
        st.session_state.original_las = None
    if 'las_df' not in st.session_state:
        st.session_state.las_df = None
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Data Configuration")
        uploaded_file = st.file_uploader("Upload LAS file", type=['las', 'LAS'])
        
        if uploaded_file:
            try:
                # Handle both text and binary LAS files
                file_contents = uploaded_file.read()
                try:
                    las_text = file_contents.decode("utf-8")
                    las = lasio.read(StringIO(las_text))
                except:
                    las = lasio.read(BytesIO(file_contents))
                
                # Store in session state
                st.session_state.original_las = las
                st.session_state.las_df = las.df()
                
                available_curves = [curve for curve in las.keys() if curve != 'DEPTH']
                
                st.header("⚙️ Model Parameters")
                epochs = st.slider("Number of Epochs", 10, 200, 50)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
                learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
                validation_split = st.slider("Validation Split (%)", 10, 30, 20) / 100
                
                st.header("📈 Curve Selection")
                target_curve = st.selectbox("Target Curve (DT)", available_curves, 
                                          index=available_curves.index('DT') if 'DT' in available_curves else 0)
                
                input_curves = []
                default_curves = ['GR', 'RT', 'PHIE', 'NPHI', 'VCL', 'RHOB', 'SW', 'DTSM']
                for i in range(8):
                    curve = st.selectbox(
                        f"Input Curve {i+1}", 
                        available_curves,
                        index=available_curves.index(default_curves[i]) if i < len(default_curves) and default_curves[i] in available_curves else 0
                    )
                    input_curves.append(curve)
                
                if st.button("🚀 Train Model", type="primary", use_container_width=True):
                    process_and_train(las, input_curves, target_curve, epochs, batch_size, 
                                    learning_rate, test_size, validation_split)
            
            except Exception as e:
                st.error(f"Error reading LAS file: {str(e)}")

    # Main content area
    if st.session_state.model_trained:
        display_results()
        
    # Data preview section
    if uploaded_file and not st.session_state.model_trained and st.session_state.las_df is not None:
        try:
            st.subheader("📋 LAS File Preview")
            st.dataframe(st.session_state.las_df.head(), use_container_width=True)
            
            st.subheader("📊 Curve Statistics")
            st.dataframe(st.session_state.las_df.describe(), use_container_width=True)
            
            st.subheader("📈 Quick View of All Curves")
            fig, ax = plt.subplots(figsize=(12, 8))
            st.session_state.las_df.plot(subplots=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Couldn't display full preview: {str(e)}")

def process_and_train(las, input_curves, target_curve, epochs, batch_size, learning_rate, test_size, validation_split):
    """Process data and train the model"""
    with st.spinner("🔄 Processing data and training model..."):
        try:
            # Extract data
            depth = las['DEPTH']
            X = np.vstack([las[curve] for curve in input_curves if curve in las.keys()]).T
            y = las[target_curve].reshape(-1, 1)
            
            if X.size == 0 or y.size == 0:
                raise ValueError("Empty data after extraction - check curve selections")
            
            # Handle missing values
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            y = np.nan_to_num(y, nan=np.nanmean(y))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            
            # Normalize
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
            
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train)
            y_test = scaler_y.transform(y_test)
            
            # Reshape for CNN
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build model
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', 
                      input_shape=(X_train.shape[1], 1)),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), 
                         loss='mean_squared_error')
            
            # Train with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,
                callbacks=[ProgressCallback()])
            
            # Predict
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test)
            
            # Store results in session state
            st.session_state.model_trained = True
            st.session_state.predictions = {
                'y_test': y_test_original,
                'y_pred': y_pred,
                'depth': depth[-len(y_test_original):],
                'X_test': scaler_X.inverse_transform(X_test.reshape(X_test.shape[0], -1)),
                'history': history,
                'r2': r2_score(y_test_original, y_pred),
                'mse': mean_squared_error(y_test_original, y_pred),
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'input_curves': input_curves,
                'target_curve': target_curve
            }
            
            st.success("✅ Model training completed!")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.session_state.model_trained = False

def display_results():
    """Display the training results and visualizations"""
    results = st.session_state.predictions
    
    if 'original_las' not in st.session_state:
        st.error("LAS file data not found. Please upload a file first.")
        return
        
    las = st.session_state.original_las
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Model Performance</h3>
            <p>R² Score: <strong>{:.4f}</strong></p>
            <p>MSE: <strong>{:.4f}</strong></p>
        </div>
        """.format(results['r2'], results['mse']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Training Summary</h3>
            <p>Final Training Loss: <strong>{:.4f}</strong></p>
            <p>Final Validation Loss: <strong>{:.4f}</strong></p>
        </div>
        """.format(
            results['history'].history['loss'][-1],
            results['history'].history['val_loss'][-1]
        ), unsafe_allow_html=True)
    
    # Training history plot
    st.subheader("📉 Training History")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(results['history'].history['loss'], label='Training Loss')
    ax1.plot(results['history'].history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    st.pyplot(fig1)
    
    # Cross plot
    st.subheader("🔄 Prediction vs Actual Cross Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    scatter = ax2.scatter(
        results['y_test'], results['y_pred'],
        c=results['X_test'][:, 6],  # Color by SW
        cmap='viridis', alpha=0.6,
        vmin=np.min(results['X_test'][:, 6]),
        vmax=np.max(results['X_test'][:, 6])
    )
    ax2.plot([results['y_test'].min(), results['y_test'].max()],
             [results['y_test'].min(), results['y_test'].max()],
             'r--', label='1:1 Line')
    ax2.set_xlabel('Actual DT')
    ax2.set_ylabel('Predicted DT')
    ax2.set_title(f'DT Prediction (R² = {results["r2"]:.3f})')
    plt.colorbar(scatter, label='SW')
    ax2.legend()
    st.pyplot(fig2)
    
    # Depth plot
    st.subheader("📏 Depth Profile Comparison")
    fig3, ax3 = plt.subplots(figsize=(8, 12))
    ax3.plot(results['y_test'], results['depth'], 'b-', label='Actual DT', linewidth=1)
    ax3.plot(results['y_pred'], results['depth'], 'r--', label='Predicted DT', linewidth=1)
    ax3.invert_yaxis()
    ax3.set_xlabel('DT (μs/ft)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('DT Comparison Along Depth')
    ax3.legend()
    st.pyplot(fig3)
    
    # Residual plot
    st.subheader("📊 Prediction Residuals")
    residuals = results['y_test'] - results['y_pred']
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax4a.hist(residuals, bins=30, edgecolor='black')
    ax4a.set_xlabel('Residual (Actual - Predicted)')
    ax4a.set_ylabel('Frequency')
    ax4a.set_title('Residual Distribution')
    
    ax4b.scatter(results['y_pred'], residuals, alpha=0.5)
    ax4b.axhline(y=0, color='r', linestyle='--')
    ax4b.set_xlabel('Predicted DT')
    ax4b.set_ylabel('Residual')
    ax4b.set_title('Residuals vs Predicted Values')
    
    st.pyplot(fig4)

    # Download Section
    st.subheader("⬇️ Export Results")
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'DEPTH': results['depth'].flatten(),
        'LOG_ACTUAL': results['y_test'].flatten(),
        'LOG_PREDICTED': results['y_pred'].flatten()
    })
    
    # Add input curves
    for i, curve in enumerate(results['input_curves']):
        df[curve] = results['X_test'][:, i]
    
    # CSV Download
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # LAS Download
    las_buffer = BytesIO()
    try:
        new_las = lasio.LASFile()
        new_las.well = st.session_state.original_las.well
        new_las.header = st.session_state.original_las.header
        
        # Add original curves
        for curve in st.session_state.original_las.curves:
            if curve.mnemonic != 'DEPTH':  # Skip depth as we're adding our own
                new_las.add_curve(
                    curve.mnemonic,
                    st.session_state.original_las[curve.mnemonic],
                    unit=curve.unit,
                    descr=curve.descr
                )
        
        # Add predicted DT
        new_las.add_curve(
            "DT_PRED",
            results['y_pred'].flatten(),
            unit="us/ft",
            descr="Predicted Sonic Log"
        )
        
        new_las.write(las_buffer, version=2.0)
        las_buffer.seek(0)
        las_download_disabled = False
    except Exception as e:
        st.warning(f"Couldn't generate LAS file: {str(e)}")
        las_download_disabled = True
    
    # Display download buttons side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📥 Download LAS",
            data=las_buffer.getvalue() if not las_download_disabled else b'',
            file_name="predictions.las",
            mime="application/octet-stream",
            disabled=las_download_disabled,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
