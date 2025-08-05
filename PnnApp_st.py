import streamlit as st
import lasio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from io import StringIO, BytesIO

tfd = tfp.distributions

# Configure the app
st.set_page_config(
    page_title="DT Prediction with PNN",
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
    .uncertainty-band {
        fill-opacity: 0.2;
    }
</style>
""", unsafe_allow_html=True)

def negative_loglikelihood(y_true, y_pred):
    """Negative log likelihood loss function for probabilistic outputs"""
    return -y_pred.log_prob(y_true)

def main():
    st.title("⛏️ Sonic Log (DT) Prediction using Probabilistic Neural Network")
    st.markdown("""
    This app predicts sonic travel time (DT) from other well log measurements using a Probabilistic Neural Network.
    The model provides both predictions and uncertainty estimates.
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
                epochs = st.slider("Number of Epochs", 10, 200, 100)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
                learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
                validation_split = st.slider("Validation Split (%)", 10, 30, 20) / 100
                n_samples = st.slider("Monte Carlo Samples", 10, 100, 50, help="Number of samples for uncertainty estimation")
                
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
                                    learning_rate, test_size, validation_split, n_samples)
            
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

def process_and_train(las, input_curves, target_curve, epochs, batch_size, learning_rate, test_size, validation_split, n_samples):
    """Process data and train the probabilistic model"""
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
            
            # Build probabilistic model
            def create_probabilistic_model(input_shape):
                model = Sequential([
                    InputLayer(input_shape=(input_shape,)),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(tfp.layers.IndependentNormal.params_size(1)),
                    tfp.layers.IndependentNormal(1)
                ])
                return model
            
            model = create_probabilistic_model(X_train.shape[1])
            model.compile(optimizer=Adam(learning_rate=learning_rate), 
                         loss=negative_loglikelihood)
            
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
            
            # Get probabilistic predictions
            y_pred_dist = model(X_test)
            y_pred_mean = y_pred_dist.mean().numpy()
            y_pred_std = y_pred_dist.stddev().numpy()
            
            # Monte Carlo sampling for uncertainty estimation
            mc_samples = np.stack([model(X_test).mean().numpy() for _ in range(n_samples)])
            y_pred_mc_mean = mc_samples.mean(axis=0)
            y_pred_mc_std = mc_samples.std(axis=0)
            
            # Inverse transform predictions
            y_pred = scaler_y.inverse_transform(y_pred_mean)
            y_pred_upper = scaler_y.inverse_transform(y_pred_mean + 1.96 * y_pred_std)
            y_pred_lower = scaler_y.inverse_transform(y_pred_mean - 1.96 * y_pred_std)
            y_test_original = scaler_y.inverse_transform(y_test)
            
            # Store results in session state
            st.session_state.model_trained = True
            st.session_state.predictions = {
                'y_test': y_test_original,
                'y_pred': y_pred,
                'y_pred_upper': y_pred_upper,
                'y_pred_lower': y_pred_lower,
                'depth': depth[-len(y_test_original):],
                'X_test': scaler_X.inverse_transform(X_test),
                'history': history,
                'r2': r2_score(y_test_original, y_pred),
                'mse': mean_squared_error(y_test_original, y_pred),
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'input_curves': input_curves,
                'target_curve': target_curve,
                'n_samples': n_samples,
                'y_pred_mc_mean': scaler_y.inverse_transform(y_pred_mc_mean),
                'y_pred_mc_std': scaler_y.inverse_transform(y_pred_mc_std)
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
            <p>Uncertainty Samples: <strong>{}</strong></p>
        </div>
        """.format(
            results['history'].history['loss'][-1],
            results['history'].history['val_loss'][-1],
            results['n_samples']
        ), unsafe_allow_html=True)
    
    # Training history plot
    st.subheader("📉 Training History")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(results['history'].history['loss'], label='Training Loss')
    ax1.plot(results['history'].history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative Log Likelihood')
    ax1.legend()
    st.pyplot(fig1)
    
    # Cross plot with uncertainty
    st.subheader("🔄 Prediction vs Actual Cross Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    # Calculate errors for errorbars
    y_err = results['y_pred_upper'] - results['y_pred']
    
    ax2.errorbar(
        results['y_test'].flatten(),
        results['y_pred'].flatten(),
        yerr=y_err.flatten(),
        fmt='o',
        alpha=0.6,
        label='Predictions with 95% CI'
    )
    ax2.plot([results['y_test'].min(), results['y_test'].max()],
             [results['y_test'].min(), results['y_test'].max()],
             'r--', label='1:1 Line')
    ax2.set_xlabel('Actual DT')
    ax2.set_ylabel('Predicted DT')
    ax2.set_title(f'DT Prediction (R² = {results["r2"]:.3f})')
    ax2.legend()
    st.pyplot(fig2)
    
    # Depth plot with uncertainty bands
    st.subheader("📏 Depth Profile Comparison with Uncertainty")
    fig3, ax3 = plt.subplots(figsize=(10, 12))
    
    # Sort by depth for proper plotting
    sort_idx = np.argsort(results['depth'].flatten())
    depth_sorted = results['depth'].flatten()[sort_idx]
    y_test_sorted = results['y_test'].flatten()[sort_idx]
    y_pred_sorted = results['y_pred'].flatten()[sort_idx]
    y_upper_sorted = results['y_pred_upper'].flatten()[sort_idx]
    y_lower_sorted = results['y_pred_lower'].flatten()[sort_idx]
    
    ax3.plot(y_test_sorted, depth_sorted, 'b-', label='Actual DT', linewidth=1)
    ax3.plot(y_pred_sorted, depth_sorted, 'r-', label='Predicted DT', linewidth=1)
    ax3.fill_betweenx(
        depth_sorted,
        y_lower_sorted,
        y_upper_sorted,
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    ax3.invert_yaxis()
    ax3.set_xlabel('DT (μs/ft)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('DT Comparison with Uncertainty Bands')
    ax3.legend()
    st.pyplot(fig3)
    
    # Uncertainty distribution
    st.subheader("📊 Prediction Uncertainty Distribution")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    relative_errors = (results['y_pred_upper'] - results['y_pred_lower']) / (2 * results['y_pred'])
    ax4.hist(relative_errors.flatten(), bins=30, edgecolor='black')
    ax4.set_xlabel('Relative Uncertainty (95% CI half-width / Prediction)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Prediction Uncertainties')
    st.pyplot(fig4)

    # Download Section
    st.subheader("⬇️ Export Results")
    
    # Create DataFrame with all data and uncertainty estimates
    df = pd.DataFrame({
        'DEPTH': results['depth'].flatten(),
        'DT_ACTUAL': results['y_test'].flatten(),
        'DT_PREDICTED': results['y_pred'].flatten(),
        'DT_PRED_UPPER': results['y_pred_upper'].flatten(),
        'DT_PRED_LOWER': results['y_pred_lower'].flatten(),
        'DT_PRED_STD': (results['y_pred_upper'].flatten() - results['y_pred'].flatten()) / 1.96,
        'DT_PRED_MC_MEAN': results['y_pred_mc_mean'].flatten(),
        'DT_PRED_MC_STD': results['y_pred_mc_std'].flatten()
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
        
        # Add predicted DT and uncertainty
        new_las.add_curve("DT_PRED", results['y_pred'].flatten(), unit="us/ft", descr="Predicted Sonic Log")
        new_las.add_curve("DT_PRED_U", results['y_pred_upper'].flatten(), unit="us/ft", descr="Upper 95% CI")
        new_las.add_curve("DT_PRED_L", results['y_pred_lower'].flatten(), unit="us/ft", descr="Lower 95% CI")
        new_las.add_curve("DT_PRED_STD", df['DT_PRED_STD'].values, unit="us/ft", descr="Standard Deviation")
        
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
            label="📥 Download CSV (Full Results)",
            data=csv_buffer.getvalue(),
            file_name="pnn_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📥 Download LAS (With Predictions)",
            data=las_buffer.getvalue() if not las_download_disabled else b'',
            file_name="pnn_predictions.las",
            mime="application/octet-stream",
            disabled=las_download_disabled,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
