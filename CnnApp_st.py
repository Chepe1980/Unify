import streamlit as st
import lasio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def dt_prediction_app():
    st.title("1D CNN for DT Prediction")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload LAS file", type=['las'])
    
    if uploaded_file is not None:
        try:
            las = lasio.read(uploaded_file)
            
            # Curve selection
            available_curves = list(las.keys())
            default_curves = ['GR', 'RT', 'PHIE', 'NPHI', 'VCL', 'RHOBMOD', 'SW', 'DTSM']
            
            with st.expander("Select Input Curves"):
                input_curves = [st.selectbox(
                    f"Select curve {i+1}", 
                    available_curves, 
                    index=available_curves.index(default_curves[i]) if default_curves[i] in available_curves else 0
                ) for i in range(8)]
                
            target_curve = st.selectbox("Select Target Curve (DT)", available_curves)
            
            # Model parameters
            with st.sidebar:
                st.header("Model Parameters")
                epochs = st.slider("Epochs", 10, 100, 50)
                batch_size = st.slider("Batch Size", 16, 128, 32)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
                
            if st.button("Run Prediction"):
                # Data preparation (similar to your existing code)
                X = np.vstack([las[curve] for curve in input_curves]).T
                y = las[target_curve].reshape(-1, 1)
                
                # ... rest of your preprocessing and modeling code ...
                
                # Display results
                st.success(f"Model trained successfully! R² Score: {r2:.4f}")
                
                # Plot results
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.scatter(y_test_original, y_pred, c=scaler_X.inverse_transform(X_test.reshape(X_test.shape[0], -1))[:, 6], cmap='jet_r')
                ax1.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], 'r--')
                ax1.set_xlabel('Real DT')
                ax1.set_ylabel('Predicted DT')
                st.pyplot(fig1)
                
                fig2, ax2 = plt.subplots(figsize=(6, 10))
                ax2.plot(y_test_original, depth[-len(y_test_original):], label='Real DT')
                ax2.plot(y_pred, depth[-len(y_test_original):], label='Predicted DT')
                ax2.invert_yaxis()
                ax2.legend()
                st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# To integrate with your main app:
# Add to your navigation options like shown in previous examples
