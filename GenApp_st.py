import streamlit as st
import lasio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from io import StringIO, BytesIO

# Configure the app
st.set_page_config(
    page_title="Sonic Log Prediction with Genetic Algorithm",
    page_icon="🧬",
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
    st.title("🧬 Sonic Log Prediction using Genetic Algorithm")
    st.markdown("""
    This app predicts sonic travel time (DT) from other well log measurements using a Genetic Algorithm.
    Upload your LAS file and configure the GA parameters in the sidebar.
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
                
                st.header("🧬 GA Parameters")
                population_size = st.slider("Population Size", 10, 200, 50)
                ngen = st.slider("Number of Generations", 10, 100, 40)
                cxpb = st.slider("Crossover Probability", 0.1, 0.9, 0.7)
                mutpb = st.slider("Mutation Probability", 0.01, 0.5, 0.2)
                tournsize = st.slider("Tournament Size", 2, 10, 3)
                
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
                
                if st.button("🧬 Run Genetic Algorithm", type="primary", use_container_width=True):
                    process_and_train(las, input_curves, target_curve, population_size, ngen, cxpb, mutpb, tournsize)
            
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

def process_and_train(las, input_curves, target_curve, population_size, ngen, cxpb, mutpb, tournsize):
    """Process data and train the GA model"""
    with st.spinner("🔄 Processing data and running Genetic Algorithm..."):
        try:
            # Extract data
            depth = las['DEPTH']
            X = np.vstack([las[curve] for curve in input_curves if curve in las.keys()]).T
            y = las[target_curve]
            
            if X.size == 0 or y.size == 0:
                raise ValueError("Empty data after extraction - check curve selections")
            
            # Handle missing values
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            y = np.nan_to_num(y, nan=np.nanmean(y))
            
            # Normalize
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # GA setup
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
            
            toolbox = base.Toolbox()
            toolbox.register("attr_float", np.random.uniform, -1, 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=X_scaled.shape[1])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def eval_dt(individual):
                dt_pred = np.sum(X_scaled * individual, axis=1)
                mse = mean_squared_error(y_scaled, dt_pred)
                return mse,
            
            toolbox.register("evaluate", eval_dt)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=tournsize)
            
            # Run GA with progress bar
            population = toolbox.population(n=population_size)
            progress_bar = st.progress(0)
            status_text = st.empty()
            history = []
            
            for gen in range(ngen):
                offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
                fits = toolbox.map(toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = toolbox.select(offspring, k=len(population))
                
                # Store best fitness
                best_ind = tools.selBest(population, k=1)[0]
                history.append(eval_dt(best_ind)[0])
                
                # Update progress
                progress = (gen + 1) / ngen
                progress_bar.progress(progress)
                status_text.text(f"Generation {gen + 1}/{ngen} - Best MSE: {history[-1]:.4f}")
            
            # Get the best individual
            best_ind = tools.selBest(population, k=1)[0]
            dt_pred_scaled = np.sum(X_scaled * best_ind, axis=1)
            dt_pred = scaler_y.inverse_transform(dt_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            r2 = r2_score(y, dt_pred)
            mse = mean_squared_error(y, dt_pred)
            
            # Store results in session state
            st.session_state.model_trained = True
            st.session_state.predictions = {
                'y_true': y,
                'y_pred': dt_pred,
                'depth': depth,
                'X': X,
                'history': history,
                'r2': r2,
                'mse': mse,
                'best_individual': best_ind,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'input_curves': input_curves,
                'target_curve': target_curve
            }
            
            st.success("✅ Genetic Algorithm completed!")
            
        except Exception as e:
            st.error(f"Error during GA execution: {str(e)}")
            st.session_state.model_trained = False

def display_results():
    """Display the GA results and visualizations"""
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
            <h3>Best Coefficients</h3>
            {", ".join([f"{coef:.4f}" for coef in results['best_individual']])}
        </div>
        """, unsafe_allow_html=True)
    
    # Training history plot
    st.subheader("📉 GA Optimization History")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(results['history'], label='Best MSE')
    ax1.set_title('Best MSE per Generation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('MSE')
    ax1.legend()
    st.pyplot(fig1)
    
    # Cross plot
    st.subheader("🔄 Prediction vs Actual Cross Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    scatter = ax2.scatter(
        results['y_true'], results['y_pred'],
        c=results['X'][:, 6],  # Color by SW
        cmap='jet_r', alpha=0.6,
        vmin=np.min(results['X'][:, 6]),
        vmax=np.max(results['X'][:, 6])
    )
    ax2.plot([results['y_true'].min(), results['y_true'].max()],
             [results['y_true'].min(), results['y_true'].max()],
             'r--', label='1:1 Line')
    ax2.set_xlabel('Actual Log (m/s)')
    ax2.set_ylabel('Predicted Log (m/s)')
    ax2.set_title(f'Log Prediction (R² = {results["r2"]:.3f})')
    plt.colorbar(scatter, label='SW')
    ax2.legend()
    st.pyplot(fig2)
    
    # Depth plot
    st.subheader("📏 Depth Profile Comparison")
    fig3, ax3 = plt.subplots(figsize=(8, 12))
    ax3.plot(results['y_true'], results['depth'], 'b-', label='Actual Log', linewidth=1)
    ax3.plot(results['y_pred'], results['depth'], 'r--', label='Predicted Log', linewidth=1)
    ax3.invert_yaxis()
    ax3.set_xlabel('DT (μs/ft)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('DT Comparison Along Depth')
    ax3.legend()
    st.pyplot(fig3)
    
    # Residual plot
    st.subheader("📊 Prediction Residuals")
    residuals = results['y_true'] - results['y_pred']
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax4a.hist(residuals, bins=30, edgecolor='black')
    ax4a.set_xlabel('Residual (Actual - Predicted)')
    ax4a.set_ylabel('Frequency')
    ax4a.set_title('Residual Distribution')
    
    ax4b.scatter(results['y_pred'], residuals, alpha=0.5)
    ax4b.axhline(y=0, color='r', linestyle='--')
    ax4b.set_xlabel('Predicted Log')
    ax4b.set_ylabel('Residual')
    ax4b.set_title('Residuals vs Predicted Values')
    
    st.pyplot(fig4)

    # Download Section
    st.subheader("⬇️ Export Results")
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'DEPTH': results['depth'],
        'Log_ACTUAL': results['y_true'],
        'Log_PREDICTED': results['y_pred'],
        'RESIDUAL': residuals
    })
    
    # Add input curves
    for i, curve in enumerate(results['input_curves']):
        df[curve] = results['X'][:, i]
    
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
            results['y_pred'],
            unit="us/ft",
            descr="Predicted Sonic Log (GA)"
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
            file_name="dt_predictions_ga.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📥 Download LAS",
            data=las_buffer.getvalue() if not las_download_disabled else b'',
            file_name="dt_predictions_ga.las",
            mime="application/octet-stream",
            disabled=las_download_disabled,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
