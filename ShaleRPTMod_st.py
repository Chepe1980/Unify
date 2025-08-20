import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import json
from streamlit.components.v1 import html
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import base64
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Shale Rock Physics Vp&Vs estimation using Xu-Payne Analysis RPT (TOC estimation Methods)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DEM and Gassmann functions
def DEM(Km, Gm, Kis, Gis, alphas, volumes):
    """
    Differential Effective Medium (DEM) model implementation
    """
    # Simplified DEM implementation for demonstration
    phi = np.sum(volumes)
    K_dry = Km * (1 - phi)**3
    G_dry = Gm * (1 - phi)**3
    return K_dry, G_dry, phi

def Gassmann_Ks(K_dry, Km, Kf, phi):
    """
    Gassmann's equation for fluid substitution
    """
    beta_dry = K_dry / (Km - K_dry)
    beta_fl = Kf / (Km - Kf)
    beta_sat = beta_dry + beta_fl
    K_sat = Km * beta_sat / (1 + beta_sat)
    return K_sat

# Improved Vernik TOC estimation functions
def vernik_toc_estimation(ip, is_val, depth, phie, clay=None):
    """
    Improved TOC estimation using Vernik's approach from Ip-Is crossplot
    with proper calibration to log response
    """
    # Normalize the inputs for better stability
    ip_norm = (ip - np.mean(ip)) / np.std(ip)
    is_norm = (is_val - np.mean(is_val)) / np.std(is_val)
    depth_norm = (depth - np.mean(depth)) / np.std(depth)
    phie_norm = (phie - np.mean(phie)) / np.std(phie)
    
    # Vernik's core relationship: TOC is inversely related to Ip and directly to Is
    # TOC = a*Ip + b*Is + c*Phi + d*Depth + constant
    toc_base = (-0.8 * ip_norm + 1.2 * is_norm - 0.5 * phie_norm - 0.3 * depth_norm + 2.0)
    
    # Convert back to original scale and apply constraints
    toc = toc_base * 2.5 + 1.0  # Scale to appropriate range
    
    # Ensure TOC is physically reasonable
    toc = np.clip(toc, 0.1, 12.0)
    
    return toc

def calibrate_toc_with_logs(ip, is_val, phie, depth, initial_toc):
    """
    Calibrate TOC estimation to better match log responses
    """
    # Create feature matrix
    X = np.column_stack([ip, is_val, phie, depth])
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use initial TOC as starting point and refine
    calibrated_toc = initial_toc.copy()
    
    # Apply corrections based on feature relationships
    ip_correction = -0.0002 * (ip - np.mean(ip))
    is_correction = 0.0003 * (is_val - np.mean(is_val))
    phi_correction = -0.5 * (phie - np.mean(phie))
    depth_correction = -0.0001 * (depth - np.mean(depth))
    
    calibrated_toc += ip_correction + is_correction + phi_correction + depth_correction
    
    return np.clip(calibrated_toc, 0.1, 12.0)

# Passey's TOC estimation method (Delta Log R method)
def passeys_toc_estimation(resistivity, sonic, depth, lom=7.5, baseline_res=1.0, baseline_sonic=90.0):
    """
    Passey's Delta Log R method for TOC estimation
    
    Parameters:
    resistivity: Formation resistivity (ohm-m)
    sonic: Sonic transit time (μs/ft)
    depth: Depth values
    lom: Level of maturity (default 7.5)
    baseline_res: Baseline resistivity in non-source rock
    baseline_sonic: Baseline sonic transit time in non-source rock
    
    Returns:
    TOC estimates using Passey's method
    """
    # Calculate Delta Log R
    delta_log_r = np.log10(resistivity / baseline_res) + 0.02 * (sonic - baseline_sonic)
    
    # Calculate TOC using Passey's equation
    toc = delta_log_r * 10**(2.297 - 0.1688 * lom)
    
    # Ensure TOC is physically reasonable
    toc = np.clip(toc, 0.1, 12.0)
    
    return toc

# Improved velocity prediction using rock physics models
def improved_velocity_prediction(toc, phi, clay, vp_measured=None, vs_measured=None):
    """
    Improved velocity prediction using calibration with measured data
    """
    # If we have measured data, calibrate the model
    if vp_measured is not None and not np.all(np.isnan(vp_measured)):
        # Calibrate using linear regression with the measured data
        valid_idx = ~np.isnan(toc) & ~np.isnan(phi) & ~np.isnan(clay) & ~np.isnan(vp_measured)
        
        if np.sum(valid_idx) > 10:  # Ensure enough data points
            X = np.column_stack([toc[valid_idx], phi[valid_idx], clay[valid_idx], np.ones(np.sum(valid_idx))])
            
            # Fit linear model for Vp
            try:
                vp_coeffs = np.linalg.lstsq(X, vp_measured[valid_idx], rcond=None)[0]
                vp_pred = vp_coeffs[0] * toc + vp_coeffs[1] * phi + vp_coeffs[2] * clay + vp_coeffs[3]
            except:
                # Fallback to empirical relationship if regression fails
                vp_pred = 4.5 - 0.12 * toc - 0.18 * phi - 0.10 * clay
        else:
            # Empirical relationship
            vp_pred = 4.5 - 0.12 * toc - 0.18 * phi - 0.10 * clay
    else:
        # Empirical relationship
        vp_pred = 4.5 - 0.12 * toc - 0.18 * phi - 0.10 * clay
    
    # For Vs prediction
    if vs_measured is not None and not np.all(np.isnan(vs_measured)):
        valid_idx = ~np.isnan(toc) & ~np.isnan(phi) & ~np.isnan(clay) & ~np.isnan(vs_measured)
        
        if np.sum(valid_idx) > 10:
            X = np.column_stack([toc[valid_idx], phi[valid_idx], clay[valid_idx], np.ones(np.sum(valid_idx))])
            
            try:
                vs_coeffs = np.linalg.lstsq(X, vs_measured[valid_idx], rcond=None)[0]
                vs_pred = vs_coeffs[0] * toc + vs_coeffs[1] * phi + vs_coeffs[2] * clay + vs_coeffs[3]
            except:
                vs_pred = vp_pred / 1.8
        else:
            vs_pred = vp_pred / 1.8
    else:
        vs_pred = vp_pred / 1.8
    
    return vp_pred, vs_pred

def vernik_density_prediction(toc, phi, clay, rhom, rhof):
    """
    Predict density using Vernik's approach for organic-rich shales
    """
    rho_organic = 1.25
    mineral_fraction = 1 - phi - toc/100
    rho_pred = mineral_fraction * rhom + phi * rhof + (toc/100) * rho_organic
    return rho_pred

# Function to add logo
def add_logo(logo_path):
    """Add a logo to the sidebar"""
    try:
        logo = Image.open(logo_path)
        st.sidebar.image(logo, use_container_width=True)
    except:
        st.sidebar.info("Logo image not found. Using text title instead.")
        st.sidebar.title("Rock Physics Analyzer")

# Help section with theory
def show_help():
    st.sidebar.header("Help & Theory")
    
    with st.sidebar.expander("Vernik's TOC Estimation Method"):
        st.write("""
        Vernik's method estimates TOC from P-impedance (Ip) and S-impedance (Is) crossplot relationships.
        
        **Key Principles:**
        - TOC is inversely related to Ip (P-impedance)
        - TOC is directly related to Is (S-impedance)
        - The relationship: TOC = a*Ip + b*Is + c*Phi + d*Depth + constant
        
        This method is particularly effective in organic-rich shales where impedance contrast is significant.
        """)
    
    with st.sidebar.expander("Passey's ΔLogR Method"):
        st.write("""
        Passey's Delta Log R method uses resistivity and sonic logs to estimate TOC.
        
        **Formula:**
        ΔLogR = log10(R/R_baseline) + 0.02*(Δt - Δt_baseline)
        TOC = ΔLogR * 10^(2.297 - 0.1688*LOM)
        
        Where:
        - R: Formation resistivity
        - R_baseline: Baseline resistivity in non-source rock
        - Δt: Sonic transit time
        - Δt_baseline: Baseline sonic transit time
        - LOM: Level of maturity (thermal maturity indicator)
        
        This method works well when resistivity and sonic logs show separation in organic-rich zones.
        """)
    
    with st.sidebar.expander("Xu-Payne Model"):
        st.write("""
        The Xu-Payne model is a rock physics model for shaly sands that:
        
        1. Uses the Differential Effective Medium (DEM) theory to model the rock frame
        2. Applies Gassmann's fluid substitution for saturation effects
        3. Accounts for clay content and its impact on elastic properties
        
        The model predicts how velocity changes with porosity, clay content, and fluid saturation.
        """)
    
    with st.sidebar.expander("How to Use This App"):
        st.write("""
        1. **Upload Data**: Prepare a CSV file with required logs (depth, porosity, Vp, etc.)
        2. **Column Mapping**: Select which columns in your data correspond to which parameters
        3. **Adjust Parameters**: Fine-tune model parameters in the sidebar
        4. **View Results**: Examine the plots and statistical metrics
        5. **Download**: Export results for further analysis
        
        For best results, ensure your data is quality-controlled and depth-matched.
        """)

def main():
    # Add logo (replace with your logo path)
    add_logo("logoToc.png")  # This will show a warning if logo.png doesn't exist
    
    st.title("Shale Rock Physics Vp&Vs estimation using Xu-Payne Analysis RPT (TOC estimation Methods)")
    
    # Show help section
    show_help()
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if not uploaded_file:
        st.warning("Please upload a CSV file")
        return

    # Process CSV file
    try:
        df = pd.read_csv(uploaded_file)
        available_columns = df.columns.tolist()
        
        # Column selection
        st.sidebar.header("Data Column Selection")
        depth_col = st.sidebar.selectbox("Depth Column", available_columns, index=0)
        por_col = st.sidebar.selectbox("Porosity Column", available_columns, index=1)
        vp_col = st.sidebar.selectbox("Vp Column (m/s)", available_columns, index=2)
        vs_col = st.sidebar.selectbox("Vs Column (m/s) - optional", [None] + available_columns, index=0)
        den_col = st.sidebar.selectbox("Density Column (g/cc) - optional", [None] + available_columns, index=0)
        sw_col = st.sidebar.selectbox("Water Saturation Column", available_columns, index=3)
        ip_col = st.sidebar.selectbox("P-Impedance (Ip) Column", available_columns, index=4)
        is_col = st.sidebar.selectbox("S-Impedance (Is) Column", available_columns, index=5)
        clay_col = st.sidebar.selectbox("Clay Volume Column - optional", [None] + available_columns, index=0)
        res_col = st.sidebar.selectbox("Resistivity Column (for Passey's method) - optional", [None] + available_columns, index=0)
        sonic_col = st.sidebar.selectbox("Sonic Column (Δt, μs/ft) - optional", [None] + available_columns, index=0)
        
        # Convert to numeric and handle missing values
        df['depth'] = pd.to_numeric(df[depth_col], errors='coerce')
        df['phie'] = pd.to_numeric(df[por_col], errors='coerce')
        df['vp'] = pd.to_numeric(df[vp_col], errors='coerce') * 0.001  # Convert m/s to km/s
        
        if vs_col and vs_col != 'None':
            df['vs'] = pd.to_numeric(df[vs_col], errors='coerce') * 0.001  # Convert m/s to km/s
        
        if den_col and den_col != 'None':
            df['den'] = pd.to_numeric(df[den_col], errors='coerce')
            
        df['sw'] = pd.to_numeric(df[sw_col], errors='coerce')
        df['ip'] = pd.to_numeric(df[ip_col], errors='coerce')
        df['is_val'] = pd.to_numeric(df[is_col], errors='coerce')
            
        if clay_col and clay_col != 'None':
            df['clay'] = pd.to_numeric(df[clay_col], errors='coerce')
        else:
            df['clay'] = 0.3  # Default value
            
        # Check if resistivity and sonic data are available
        passey_available = False
        if res_col and res_col != 'None' and sonic_col and sonic_col != 'None':
            df['resistivity'] = pd.to_numeric(df[res_col], errors='coerce')
            df['sonic'] = pd.to_numeric(df[sonic_col], errors='coerce')
            
            # Check if we have valid data for Passey's method
            if not df['resistivity'].isna().all() and not df['sonic'].isna().all():
                passey_available = True
                st.sidebar.success("Resistivity and Sonic data found. Passey's method will be used.")
            else:
                st.sidebar.warning("Resistivity or Sonic data contains missing values. Passey's method may not work properly.")
        else:
            st.sidebar.info("Resistivity and/or Sonic data not selected. Only Vernik's method will be used.")
        
        # Handle missing values
        required_cols = ['depth', 'phie', 'vp', 'sw', 'ip', 'is_val']
        df = df.dropna(subset=required_cols)
        df['index'] = range(len(df))
        df = df.sort_values('depth')
        
        st.subheader("Sample Data")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return

    # Model parameters
    st.sidebar.header("Model Parameters")
    Km_lim = st.sidebar.number_input("Limestone Bulk Modulus (GPa)", value=77.0)
    Gm_lim = st.sidebar.number_input("Limestone Shear Modulus (GPa)", value=32.0)
    rhom_lim = st.sidebar.number_input("Limestone Density (g/cm3)", value=2.71)
    Kf = st.sidebar.number_input("Fluid Bulk Modulus (GPa)", value=2.24)
    rhof = st.sidebar.number_input("Fluid Density (g/cm3)", value=0.94)
    phimax = st.sidebar.number_input("Maximum Porosity", value=0.4)
    
    # TOC estimation parameters
    st.sidebar.header("TOC Estimation Parameters")
    
    # Vernik method parameters
    st.sidebar.subheader("Vernik Method Parameters")
    ip_weight = st.sidebar.slider("Ip weight in TOC estimation", -2.0, 2.0, -0.8)
    is_weight = st.sidebar.slider("Is weight in TOC estimation", -2.0, 2.0, 1.2)
    phi_weight = st.sidebar.slider("Porosity weight in TOC estimation", -2.0, 2.0, -0.5)
    depth_weight = st.sidebar.slider("Depth weight in TOC estimation", -2.0, 2.0, -0.3)
    base_toc = st.sidebar.slider("Base TOC level", 0.0, 5.0, 2.0)
    
    # Passey method parameters
    st.sidebar.subheader("Passey Method Parameters")
    lom = st.sidebar.slider("Level of Maturity (LOM)", 6.0, 12.0, 7.5)
    baseline_res = st.sidebar.number_input("Baseline Resistivity (ohm-m)", value=1.0)
    baseline_sonic = st.sidebar.number_input("Baseline Sonic (μs/ft)", value=90.0)

    # Generate Xu-Payne curves
    def generate_xu_payne_curves():
        phi_range = np.linspace(0, phimax, 50)
        vp_range = 5.0 - 8.0 * phi_range
        return [(phi_range, vp_range)]

    # Improved TOC estimation with proper Ip-Is relationship
    def estimate_toc(df):
        # First pass estimation using Vernik's approach
        initial_toc = vernik_toc_estimation(
            df['ip'].values, 
            df['is_val'].values, 
            df['depth'].values,
            df['phie'].values
        )
        
        # Calibrate with log responses
        calibrated_toc = calibrate_toc_with_logs(
            df['ip'].values,
            df['is_val'].values,
            df['phie'].values,
            df['depth'].values,
            initial_toc
        )
        
        # Apply user-defined weights
        ip_norm = (df['ip'] - df['ip'].mean()) / df['ip'].std()
        is_norm = (df['is_val'] - df['is_val'].mean()) / df['is_val'].std()
        phi_norm = (df['phie'] - df['phie'].mean()) / df['phie'].std()
        depth_norm = (df['depth'] - df['depth'].mean()) / df['depth'].std()
        
        final_toc = (ip_weight * ip_norm + 
                    is_weight * is_norm + 
                    phi_weight * phi_norm + 
                    depth_weight * depth_norm + base_toc)
        
        # Ensure positive values and reasonable range
        final_toc = np.clip(final_toc, 0.1, 12.0)
        
        return final_toc

    # Predict properties
    def predict_properties(df, toc):
        clay = df['clay'].values
        vp_measured = df['vp'].values if 'vp' in df.columns else None
        vs_measured = df['vs'].values if 'vs' in df.columns else None
        
        vp_pred, vs_pred = improved_velocity_prediction(
            toc, df['phie'].values, clay, vp_measured, vs_measured
        )
        
        rho_pred = vernik_density_prediction(toc, df['phie'].values, clay, rhom_lim, rhof)
        
        return vp_pred, vs_pred, rho_pred

    # Estimate TOC using both methods
    df['toc_vernik'] = estimate_toc(df)
    
    # Estimate TOC using Passey's method if we have the required data
    if passey_available:
        # Filter out NaN values for Passey's method
        valid_idx = ~df['resistivity'].isna() & ~df['sonic'].isna()
        df_valid = df[valid_idx].copy()
        
        if len(df_valid) > 0:
            df.loc[valid_idx, 'toc_passey'] = passeys_toc_estimation(
                df_valid['resistivity'].values, 
                df_valid['sonic'].values, 
                df_valid['depth'].values,
                lom, 
                baseline_res, 
                baseline_sonic
            )
            
            # Fill NaN values with Vernik's TOC for consistency
            df['toc_passey'] = df['toc_passey'].fillna(df['toc_vernik'])
        else:
            st.warning("No valid data for Passey's method after filtering NaN values. Using Vernik's method.")
            df['toc_passey'] = df['toc_vernik']
    else:
        df['toc_passey'] = df['toc_vernik']  # Default to Vernik's values
    
    # Use Vernik's TOC for property prediction
    df['vp_pred'], df['vs_pred'], df['den_pred'] = predict_properties(df, df['toc_vernik'])
    
    # Calculate errors
    df['vp_error'] = df['vp_pred'] - df['vp']
    vp_rmse = np.sqrt(np.mean(df['vp_error']**2))
    vp_mae = np.mean(np.abs(df['vp_error']))
    vp_r2 = 1 - np.sum(df['vp_error']**2) / np.sum((df['vp'] - df['vp'].mean())**2)
    
    if 'vs' in df.columns and not df['vs'].isna().all():
        df['vs_error'] = df['vs_pred'] - df['vs']
        vs_rmse = np.sqrt(np.mean(df['vs_error']**2))
        vs_mae = np.mean(np.abs(df['vs_error']))
        vs_r2 = 1 - np.sum(df['vs_error']**2) / np.sum((df['vs'] - df['vs'].mean())**2)
    
    if 'den' in df.columns and not df['den'].isna().all():
        df['den_error'] = df['den_pred'] - df['den']
        den_rmse = np.sqrt(np.mean(df['den_error']**2))
        den_mae = np.mean(np.abs(df['den_error']))

    # Create main interactive plots
    def create_main_plots():
        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=("Vp vs Depth", "Ip vs Is colored by TOC", "TOC vs Depth", "TOC vs Ip/Is Relationship"),
            specs=[[{}, {}], [{}, {}]]
        )

        # Vp vs Depth
        fig.add_trace(
            go.Scatter(x=df['vp'], y=df['depth'], mode='markers', name='Measured Vp',
                      marker=dict(color='blue', size=6, opacity=0.7)), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['vp_pred'], y=df['depth'], mode='markers', name='Predicted Vp',
                      marker=dict(color='red', size=6, opacity=0.7)), row=1, col=1
        )

        # Ip vs Is colored by TOC
        fig.add_trace(
            go.Scatter(x=df['ip'], y=df['is_val'], mode='markers', name='Ip vs Is',
                      marker=dict(color=df['toc_vernik'], colorscale='Reds', showscale=True,
                                 colorbar=dict(title="TOC (%)", thickness=10, len=0.5, x=0.82, y=0.5), size=6)), row=1, col=2
        )

        # TOC vs Depth - both methods if available
        fig.add_trace(
            go.Scatter(x=df['toc_vernik'], y=df['depth'], mode='markers', name='Vernik TOC',
                      marker=dict(color='blue', size=6, opacity=0.7)), row=2, col=1
        )
        
        if passey_available:
            fig.add_trace(
                go.Scatter(x=df['toc_passey'], y=df['depth'], mode='markers', name='Passey TOC',
                          marker=dict(color='green', size=6, opacity=0.7)), row=2, col=1
            )

        # TOC vs Ip/Is relationship
        fig.add_trace(
            go.Scatter(x=df['ip']/df['is_val'], y=df['toc_vernik'], mode='markers', name='Ip/Is vs TOC',
                      marker=dict(color=df['depth'], colorscale='Viridis', showscale=True,
                                 colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5), size=6)), row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True)
        fig.update_yaxes(title_text="Depth", row=1, col=1, autorange="reversed")
        fig.update_xaxes(title_text="Vp (km/s)", row=1, col=1)
        fig.update_xaxes(title_text="Ip (m/s*g/cc)", row=1, col=2)
        fig.update_yaxes(title_text="Is (m/s*g/cc)", row=1, col=2)
        fig.update_xaxes(title_text="TOC (%)", row=2, col=1)
        fig.update_yaxes(title_text="Depth", row=2, col=1, autorange="reversed")
        fig.update_xaxes(title_text="Ip/Is Ratio", row=2, col=2)
        fig.update_yaxes(title_text="TOC (%)", row=2, col=2)
        
        return fig

    # Create TOC comparison plots
    def create_toc_comparison_plots():
        if not passey_available:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("TOC Comparison", "TOC Difference vs Depth", 
                           "Vernik vs Passey TOC Crossplot", "TOC Ratio vs Depth"),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # TOC Comparison
        fig.add_trace(
            go.Scatter(x=df['toc_vernik'], y=df['depth'], mode='markers', name='Vernik TOC',
                      marker=dict(color='blue', size=6, opacity=0.7)), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['toc_passey'], y=df['depth'], mode='markers', name='Passey TOC',
                      marker=dict(color='green', size=6, opacity=0.7)), row=1, col=1
        )
        
        # TOC Difference vs Depth
        toc_diff = df['toc_vernik'] - df['toc_passey']
        fig.add_trace(
            go.Scatter(x=toc_diff, y=df['depth'], mode='markers', name='TOC Difference',
                      marker=dict(color=toc_diff, colorscale='RdBu', showscale=True,
                                 colorbar=dict(title="TOC Difference", thickness=10, len=0.5, x=0.82, y=0.5), size=6)), row=1, col=2
        )
        
        # Vernik vs Passey TOC Crossplot
        fig.add_trace(
            go.Scatter(x=df['toc_vernik'], y=df['toc_passey'], mode='markers', name='TOC Crossplot',
                      marker=dict(color=df['depth'], colorscale='Viridis', showscale=True,
                                 colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5), size=6)), row=2, col=1
        )
        
        # Add 1:1 line for reference
        min_toc = min(df['toc_vernik'].min(), df['toc_passey'].min())
        max_toc = max(df['toc_vernik'].max(), df['toc_passey'].max())
        fig.add_trace(
            go.Scatter(x=[min_toc, max_toc], y=[min_toc, max_toc], mode='lines', 
                      name='1:1 Line', line=dict(color='red', dash='dash')), row=2, col=1
        )
        
        # TOC Ratio vs Depth
        toc_ratio = df['toc_vernik'] / df['toc_passey']
        fig.add_trace(
            go.Scatter(x=toc_ratio, y=df['depth'], mode='markers', name='TOC Ratio',
                      marker=dict(color=df['depth'], colorscale='Viridis', showscale=True,
                                 colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5), size=6)), row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_yaxes(title_text="Depth", row=1, col=1, autorange="reversed")
        fig.update_xaxes(title_text="TOC (%)", row=1, col=1)
        fig.update_xaxes(title_text="TOC Difference (Vernik - Passey)", row=1, col=2)
        fig.update_yaxes(title_text="Depth", row=1, col=2, autorange="reversed")
        fig.update_xaxes(title_text="Vernik TOC (%)", row=2, col=1)
        fig.update_yaxes(title_text="Passey TOC (%)", row=2, col=1)
        fig.update_xaxes(title_text="TOC Ratio (Vernik/Passey)", row=2, col=2)
        fig.update_yaxes(title_text="Depth", row=2, col=2, autorange="reversed")
        
        return fig

    # Create TOC relationship analysis plots with Plotly
    def create_toc_relationship_plots():
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("TOC vs Ip", "TOC vs Is", "TOC vs Ip/Is Ratio", "TOC vs Porosity"),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # TOC vs Ip
        fig.add_trace(
            go.Scatter(
                x=df['ip'], y=df['toc_vernik'], mode='markers', name='TOC vs Ip',
                marker=dict(color=df['depth'], colorscale='Viridis', showscale=True,
                           colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5), size=6),
                hovertemplate='Ip: %{x:.0f}<br>TOC: %{y:.2f}%<br>Depth: %{marker.color:.0f}<extra></extra>'
            ), row=1, col=1
        )
        
        # TOC vs Is
        fig.add_trace(
            go.Scatter(
                x=df['is_val'], y=df['toc_vernik'], mode='markers', name='TOC vs Is',
                marker=dict(color=df['depth'], colorscale='Viridis', showscale=True,
                           colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5), size=6),
                hovertemplate='Is: %{x:.0f}<br>TOC: %{y:.2f}%<br>Depth: %{marker.color:.0f><extra></extra>'
            ), row=1, col=2
        )
        
        # TOC vs Ip/Is Ratio
        fig.add_trace(
            go.Scatter(
                x=df['ip']/df['is_val'], y=df['toc_vernik'], mode='markers', name='TOC vs Ip/Is',
                marker=dict(color=df['depth'], colorscale='Viridis', showscale=True,
                           colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5), size=6),
                hovertemplate='Ip/Is: %{x:.2f}<br>TOC: %{y:.2f}%<br>Depth: %{marker.color:.0f}<extra></extra>'
            ), row=2, col=1
        )
        
        # TOC vs Porosity
        fig.add_trace(
            go.Scatter(
                x=df['phie'], y=df['toc_vernik'], mode='markers', name='TOC vs Porosity',
                marker=dict(color=df['depth'], colorscale='Viridis', showscale=True,
                           colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5), size=6),
                hovertemplate='Porosity: %{x:.3f}<br>TOC: %{y:.2f}%<br>Depth: %{marker.color:.0f}<extra></extra>'
            ), row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="TOC Relationship Analysis")
        fig.update_xaxes(title_text="Ip (m/s*g/cc)", row=1, col=1)
        fig.update_yaxes(title_text="TOC (%)", row=1, col=1)
        fig.update_xaxes(title_text="Is (m/s*g/cc)", row=1, col=2)
        fig.update_yaxes(title_text="TOC (%)", row=1, col=2)
        fig.update_xaxes(title_text="Ip/Is Ratio", row=2, col=1)
        fig.update_yaxes(title_text="TOC (%)", row=2, col=1)
        fig.update_xaxes(title_text="Porosity", row=2, col=2)
        fig.update_yaxes(title_text="TOC (%)", row=2, col=2)
        
        return fig

    # Create TOC vs Depth plot with Plotly
    def create_toc_depth_plot():
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['toc_vernik'], y=df['depth'], mode='markers',
                marker=dict(
                    color=df['toc_vernik'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="TOC (%)", thickness=10, len=0.5, x=0.82, y=0.5),
                    size=6,
                    opacity=0.8
                ),
                hovertemplate='TOC: %{x:.2f}%<br>Depth: %{y:.0f}<extra></extra>',
                name='Vernik TOC'
            )
        )
        
        if passey_available:
            fig.add_trace(
                go.Scatter(
                    x=df['toc_passey'], y=df['depth'], mode='markers',
                    marker=dict(
                        color=df['toc_passey'],
                        colorscale='Greens',
                        showscale=True,
                        colorbar=dict(title="Passey TOC (%)", thickness=10, len=0.5, x=0.82, y=0.5),
                        size=6,
                        opacity=0.8
                    ),
                    hovertemplate='Passey TOC: %{x:.2f}%<br>Depth: %{y:.0f}<extra></extra>',
                    name='Passey TOC'
                )
            )
        
        fig.update_layout(
            title="TOC vs Depth",
            xaxis_title="TOC (%)",
            yaxis_title="Depth",
            height=500,
            showlegend=True
        )
        fig.update_yaxes(autorange="reversed")  # Deeper at bottom
        
        return fig

    # Create TOC comparison scatter plot
    def create_toc_comparison_scatter():
        if not passey_available:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['toc_vernik'], y=df['toc_passey'], mode='markers',
                marker=dict(
                    color=df['depth'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Depth", thickness=10, len=0.5, x=0.82, y=0.5),
                    size=6,
                    opacity=0.7
                ),
                hovertemplate='Vernik TOC: %{x:.2f}%<br>Passey TOC: %{y:.2f}%<br>Depth: %{marker.color:.0f}<extra></extra>',
                name='TOC Comparison'
            )
        )
        
        # Add 1:1 line
        min_toc = min(df['toc_vernik'].min(), df['toc_passey'].min())
        max_toc = max(df['toc_vernik'].max(), df['toc_passey'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_toc, max_toc], y=[min_toc, max_toc], 
                mode='lines', line=dict(color='red', dash='dash'),
                name='1:1 Line'
            )
        )
        
        fig.update_layout(
            title="Vernik vs Passey TOC Comparison",
            xaxis_title="Vernik TOC (%)",
            yaxis_title="Passey TOC (%)",
            height=600,
            showlegend=True
        )
        
        return fig

    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Main Results", "TOC Relationships", "TOC Comparison", "Statistics", "Data"])
    
    with tab1:
        st.header("Main Results")
        st.plotly_chart(create_main_plots(), use_container_width=True)
        
    with tab2:
        st.header("TOC Relationship Analysis")
        st.plotly_chart(create_toc_relationship_plots(), use_container_width=True)
        st.plotly_chart(create_toc_depth_plot(), use_container_width=True)
        
    with tab3:
        st.header("TOC Method Comparison")
        if passey_available:
            st.plotly_chart(create_toc_comparison_plots(), use_container_width=True)
            st.plotly_chart(create_toc_comparison_scatter(), use_container_width=True)
        else:
            st.info("Passey's method requires resistivity and sonic data. No comparison available.")
            
    with tab4:
        st.header("Prediction Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vp RMSE", f"{vp_rmse:.3f} km/s")
            st.metric("Vp MAE", f"{vp_mae:.3f} km/s")
            st.metric("Vp R²", f"{vp_r2:.3f}")
        
        with col2:
            if 'vs' in df.columns and not df['vs'].isna().all():
                st.metric("Vs RMSE", f"{vs_rmse:.3f} km/s")
                st.metric("Vs MAE", f"{vs_mae:.3f} km/s")
                st.metric("Vs R²", f"{vs_r2:.3f}")
            else:
                st.info("Vs data not available")
        
        with col3:
            if 'den' in df.columns and not df['den'].isna().all():
                st.metric("Density RMSE", f"{den_rmse:.3f} g/cc")
                st.metric("Density MAE", f"{den_mae:.3f} g/cc")
            else:
                st.info("Density data not available")
                
        # TOC comparison statistics if Passey's method is available
        if passey_available:
            st.subheader("TOC Method Comparison Statistics")
            toc_diff = df['toc_vernik'] - df['toc_passey']
            toc_corr = df['toc_vernik'].corr(df['toc_passey'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean TOC Difference", f"{toc_diff.mean():.3f} %")
            with col2:
                st.metric("TOC Correlation", f"{toc_corr:.3f}")
            with col3:
                st.metric("RMS TOC Difference", f"{np.sqrt(np.mean(toc_diff**2)):.3f} %")
            with col4:
                st.metric("Max TOC Difference", f"{toc_diff.abs().max():.3f} %")
                
        # Show correlation matrix
        st.subheader("Correlation Matrix")
        corr_cols = ['toc_vernik', 'ip', 'is_val', 'phie', 'vp', 'depth']
        if passey_available:
            corr_cols.append('toc_passey')
        corr_matrix = df[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45)
        ax.set_yticklabels(corr_cols)
        
        # Add correlation values to cells
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
    with tab5:
        st.header("Detailed Results with TOC Estimates")
        display_cols = ['depth', 'phie', 'vp', 'vp_pred', 'vp_error', 'ip', 'is_val', 'toc_vernik']
        if passey_available:
            display_cols.append('toc_passey')
        st.dataframe(df[display_cols].head(20))
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button("Download Results", data=csv, file_name="accurate_toc_results.csv", mime="text/csv")

if __name__ == '__main__':
    main()
