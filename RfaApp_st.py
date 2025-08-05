import streamlit as st
import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from io import StringIO, BytesIO
import seaborn as sns

# Configure the app
st.set_page_config(
    page_title="Sonic Log Prediction with Random Forest",
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
    st.title("⛏️ Sonic Log Prediction using Random Forest")
    st.markdown("""
    This app predicts sonic travel time (DT) from other well log measurements using a Random Forest regressor.
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
                st.session_state.las_df.reset_index(inplace=True)
                
                available_curves = [curve for curve in las.keys() if curve != 'DEPTH']
                
                st.header("⚙️ Model Parameters")
                n_estimators = st.slider("Number of Trees", 10, 500, 100)
                max_depth = st.slider("Max Depth", 1, 30, 10)
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
                random_state = st.number_input("Random State", 0, 100, 42)
                
                st.header("📈 Curve Selection")
                target_curve = st.selectbox("Target Curve (DT)", available_curves, 
                                          index=available_curves.index('DT') if 'DT' in available_curves else 0)
                
                # Default input curves
                default_curves = ['GR', 'RT', 'PHIE', 'NPHI', 'VCL', 'RHOBMOD', 'SW', 'DTSM']
                input_curves = []
                
                for i in range(8):
                    curve = st.selectbox(
                        f"Input Curve {i+1}", 
                        available_curves,
                        index=available_curves.index(default_curves[i]) if i < len(default_curves) and default_curves[i] in available_curves else 0
                    )
                    input_curves.append(curve)
                
                if st.button("🚀 Train Model", type="primary", use_container_width=True):
                    process_and_train(las, input_curves, target_curve, n_estimators, 
                                    max_depth, test_size, random_state)
            
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
            st.session_state.las_df.set_index('DEPTH').plot(subplots=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Couldn't display full preview: {str(e)}")

def process_and_train(las, input_curves, target_curve, n_estimators, max_depth, test_size, random_state):
    """Process data and train the model"""
    with st.spinner("🔄 Processing data and training model..."):
        try:
            # Extract data from DataFrame in session state
            df = st.session_state.las_df.copy()
            
            # Drop rows with missing values in selected curves
            df = df.dropna(subset=input_curves + [target_curve])
            
            if df.empty:
                raise ValueError("No data remaining after dropping missing values - check curve selections")
            
            # Prepare features and target
            X = df[input_curves].values
            y = df[target_curve].values
            depth = df['DEPTH'].values
            
            # Split data (maintaining depth order)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            depth_test = depth[split_idx:]
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Store results in session state
            st.session_state.model_trained = True
            st.session_state.predictions = {
                'y_test': y_test,
                'y_pred': y_pred,
                'depth': depth_test,
                'X_test': X_test,
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'model': model,
                'input_curves': input_curves,
                'target_curve': target_curve,
                'feature_importances': model.feature_importances_
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
        
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model Performance</h3>
            <p>R² Score: <strong>{results['r2']:.4f}</strong></p>
            <p>MSE: <strong>{results['mse']:.4f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Feature Importances</h3>
            {''.join([f'<p>{curve}: <strong>{imp:.4f}</strong></p>' 
                     for curve, imp in zip(results['input_curves'], results['feature_importances'])])}
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance plot
    st.subheader("📊 Feature Importances")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    importances = pd.Series(results['feature_importances'], index=results['input_curves'])
    importances.sort_values().plot(kind='barh', ax=ax1)
    ax1.set_title('Feature Importances from Random Forest')
    ax1.set_xlabel('Relative Importance')
    st.pyplot(fig1)
    
    # Cross plot
    st.subheader("🔄 Prediction vs Actual Cross Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    # Find SW index for coloring
    sw_index = next((i for i, curve in enumerate(results['input_curves']) if curve == 'SW'), None)
    color_data = results['X_test'][:, sw_index] if sw_index is not None else None
    
    if color_data is not None:
        scatter = ax2.scatter(
            results['y_test'], results['y_pred'],
            c=color_data,
            cmap='jet_r', alpha=0.6,
            vmin=np.min(color_data),
            vmax=np.max(color_data)
        )
        plt.colorbar(scatter, label='SW')
    else:
        ax2.scatter(results['y_test'], results['y_pred'], alpha=0.6)
    
    ax2.plot([results['y_test'].min(), results['y_test'].max()],
             [results['y_test'].min(), results['y_test'].max()],
             'r--', label='1:1 Line')
    ax2.set_xlabel('Actual Log (m/s)')
    ax2.set_ylabel('Predicted Log (m/s)')
    ax2.set_title(f'Log Prediction (R² = {results["r2"]:.3f})')
    ax2.legend()
    st.pyplot(fig2)
    
    # Depth plot
    st.subheader("📏 Depth Profile Comparison")
    fig3, ax3 = plt.subplots(figsize=(8, 12))
    ax3.plot(results['y_test'], results['depth'], 'b-', label='Actual Log', linewidth=1)
    ax3.plot(results['y_pred'], results['depth'], 'r--', label='Predicted Log', linewidth=1)
    ax3.invert_yaxis()
    ax3.set_xlabel('Target Log (m/s)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('LogComparison Along Depth')
    ax3.legend()
    st.pyplot(fig3)
    
    # Hexbin plot
    st.subheader("📈 Density Plot of Predictions")
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    hb = ax4.hexbin(results['y_test'], results['y_pred'], gridsize=50, cmap='jet_r', bins='log')
    plt.colorbar(hb, label='Log10 Count')
    ax4.plot([results['y_test'].min(), results['y_test'].max()],
             [results['y_test'].min(), results['y_test'].max()],
             'r--', label='1:1 Line')
    ax4.set_xlabel('Actual Log')
    ax4.set_ylabel('Predicted Log')
    ax4.set_title(f'Real vs Predicted Log (R² = {results["r2"]:.4f})')
    ax4.legend()
    st.pyplot(fig4)
    
    # Pairplot for selected features
    st.subheader("📊 Feature Relationships")
    plot_curves = results['input_curves'][:4]  # Show first 4 curves to avoid overcrowding
    df_plot = pd.DataFrame(results['X_test'], columns=results['input_curves'])
    df_plot['DT_ACTUAL'] = results['y_test']
    df_plot['DT_PRED'] = results['y_pred']
    
    # Bin DT values for coloring
    bins = np.linspace(min(results['y_test']), max(results['y_test']), 4)
    df_plot['DT_BIN'] = np.digitize(results['y_test'], bins)
    
    fig5 = sns.pairplot(df_plot, vars=plot_curves, hue='DT_BIN', palette='viridis', diag_kind='hist')
    st.pyplot(fig5)

    # Download Section
    st.subheader("⬇️ Export Results")
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'DEPTH': results['depth'],
        f'{results["target_curve"]}_ACTUAL': results['y_test'],
        f'{results["target_curve"]}_PREDICTED': results['y_pred'],
        'RESIDUAL': results['y_test'] - results['y_pred']
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
        
        # Add predicted DT (need to align with original depth)
        full_depth = st.session_state.original_las['DEPTH']
        full_pred = np.full_like(full_depth, np.nan)
        test_indices = np.isin(full_depth, results['depth'])
        full_pred[test_indices] = results['y_pred']
        
        new_las.add_curve(
            f"{results['target_curve']}_PRED",
            full_pred,
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
            file_name="Log_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📥 Download LAS",
            data=las_buffer.getvalue() if not las_download_disabled else b'',
            file_name="dt_predictions.las",
            mime="application/octet-stream",
            disabled=las_download_disabled,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
