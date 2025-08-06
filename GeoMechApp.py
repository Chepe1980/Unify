import streamlit as st
import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO

# Configure the app
st.set_page_config(
    page_title="Geomechanical Analysis & Anisotropy Calculator",
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

def calculate_poissons_ratio(vp, vs):
    """Calculate Poisson's ratio from Vp and Vs"""
    return (vp**2 - 2*vs**2) / (2*(vp**2 - vs**2))

def calculate_youngs_modulus(vp, vs, rho):
    """
    Calculate Young's Modulus (E) in GPa
    Formula: E = ρVs²(3Vp² - 4Vs²)/(Vp² - Vs²)
    """
    return (rho * vs**2 * (3*vp**2 - 4*vs**2) / (vp**2 - vs**2)) / 1e9

def normalize(x):
    """Min-max normalization to [0,1] range"""
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def main():
    st.title("⛏️ Geomechanical Analysis & Anisotropy Calculator")
    st.markdown("""
    This app calculates geomechanical properties and anisotropy parameters from well log data.
    Upload your LAS file containing sonic and density logs to get started.
    """)
    
    # Initialize session state
    if 'las_data' not in st.session_state:
        st.session_state.las_data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
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
                st.session_state.las_data = las
                st.session_state.las_df = las.df()
                
                st.success("LAS file loaded successfully!")
                
                # Curve selection
                st.header("🔧 Parameter Configuration")
                
                # Try to automatically detect curve names
                available_curves = [curve for curve in las.keys() if curve != 'DEPTH']
                
                # Vp curve selection
                vp_options = ['DTPMOD', 'VP', 'VPC', 'DTCO']
                vp_curve = st.selectbox(
                    "Compressional Wave Velocity (Vp)",
                    options=available_curves,
                    index=next((i for i, curve in enumerate(available_curves) if curve in vp_options), 0))
                
                # Vs curve selection
                vs_options = ['DTSMOD', 'VS', 'VSC', 'DTSM']
                vs_curve = st.selectbox(
                    "Shear Wave Velocity (Vs)",
                    options=available_curves,
                    index=next((i for i, curve in enumerate(available_curves) if curve in vs_options), 0))
                
                # Density curve selection
                rho_options = ['RHOBMOD', 'RHO', 'RHOB', 'DEN']
                rho_curve = st.selectbox(
                    "Bulk Density (RHOB)",
                    options=available_curves,
                    index=next((i for i, curve in enumerate(available_curves) if curve in rho_options), 0))
                
                if st.button("🚀 Calculate Geomechanical Properties", type="primary", use_container_width=True):
                    calculate_properties(las, vp_curve, vs_curve, rho_curve)
            
            except Exception as e:
                st.error(f"Error reading LAS file: {str(e)}")

    # Main content area
    if st.session_state.results:
        display_results()
        
    # Data preview section
    if uploaded_file and not st.session_state.results and st.session_state.las_df is not None:
        try:
            st.subheader("📋 LAS File Preview")
            st.dataframe(st.session_state.las_df.head(), use_container_width=True)
            
            st.subheader("📊 Curve Statistics")
            st.dataframe(st.session_state.las_df.describe(), use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't display full preview: {str(e)}")

def calculate_properties(las, vp_curve, vs_curve, rho_curve):
    """Calculate all geomechanical properties"""
    with st.spinner("🔄 Calculating geomechanical properties..."):
        try:
            # Extract velocities and density
            try:
                # Case 1: If slowness logs are available (DTPMOD = compressional, DTSMOD = shear)
                Vp = 304800 / las[vp_curve]    # Convert μs/ft to m/s
                Vs = 304800 / las[vs_curve]    # Convert μs/ft to m/s
            except:
                # Case 2: If velocity logs are available (VP = compressional, VS = shear)
                Vp = las[vp_curve]
                Vs = las[vs_curve]
            
            Rho = las[rho_curve] * 1000  # Convert to kg/m³

            # Calculate Poisson's ratio (with safety checks)
            valid_velocity_mask = (Vp > 0) & (Vs > 0) & (Vp > Vs*np.sqrt(2))
            PR = np.full_like(Vp, np.nan)  # Initialize with NaNs
            PR[valid_velocity_mask] = calculate_poissons_ratio(Vp[valid_velocity_mask], Vs[valid_velocity_mask])

            # Calculate Young's Modulus (with safety checks)
            YM = np.full_like(Vp, np.nan)
            YM[valid_velocity_mask] = calculate_youngs_modulus(Vp[valid_velocity_mask],
                                                             Vs[valid_velocity_mask],
                                                             Rho[valid_velocity_mask])

            # Calculate Vp/Vs ratio (with safety checks)
            Vp_Vs = np.divide(Vp, Vs, out=np.zeros_like(Vp), where=(Vs != 0))

            # Calculate Thomsen parameters
            with np.errstate(divide='ignore', invalid='ignore'):
                delta = ((1 + 3.87 * Vp_Vs - 5.54)**2 - (Vp_Vs**2 - 1)**2) / (2 * Vp_Vs**2 * (Vp_Vs**2 - 1))
                epsilon = 0.2090 * Vp_Vs - 0.2397
                gamma = 0.4014 * Vp_Vs - 0.5576
                g = (Vs**2/Vp**2)
                term1 = (delta/(1-2*g))
                term2 = ((epsilon-gamma*g**2)/(1-g**2))
                term3 = (epsilon/((2*g**2)*(1-2*g**2)*(1-g**2)))
                term4 = (2*gamma/(1-2*g**2))
                term5 = (delta/(1-g**2))
                term6 = (4*PR*delta)
                term7 = ((4*PR**2)*(epsilon-g*gamma))

                term8 = (-1*epsilon)*((1-2*g)*delta)
                term9 = (2*g*(3-4*g)*(1-g))
                term10 = 4*(1-2*g)*gamma
                term11 = (3-4*g)

                # Calculate Poisson's Ratio Vertical and Horizontal
                Vv = PR*(1+term1-term2)
                Vh = PR*(1+term3-term4-term5)

                # Calculate Young's Modulus Vertical and Horizontal
                YMv = (YM - term6 + term7)
                YMh = (YM*(1+(term8/term9)+(term10/term11)))

            # Normalize the parameters (0-1 scaling)
            YMv_norm = np.full_like(YMv, np.nan)
            YMh_norm = np.full_like(YMh, np.nan)
            Vv_norm = np.full_like(Vv, np.nan)
            Vh_norm = np.full_like(Vh, np.nan)

            valid_mask = ~np.isnan(YMv)  # Use same mask for all normalized parameters
            YMv_norm[valid_mask] = normalize(YMv[valid_mask])
            YMh_norm[valid_mask] = normalize(YMh[valid_mask])
            Vv_norm[valid_mask] = normalize(Vv[valid_mask])
            Vh_norm[valid_mask] = normalize(Vh[valid_mask])

            BRITv = (YMv_norm + Vv_norm)/2
            BRITh = (YMh_norm + Vh_norm)/2

            # Store results in session state
            st.session_state.results = {
                'PR': PR,
                'YM': YM,
                'delta': delta,
                'epsilon': epsilon,
                'gamma': gamma,
                'Vv': Vv,
                'Vh': Vh,
                'YMv': YMv,
                'YMh': YMh,
                'BRITv': BRITv,
                'BRITh': BRITh,
                'depth': las.index,
                'valid_mask': valid_mask,
                'Vp': Vp,
                'Vs': Vs,
                'Rho': Rho/1000  # Convert back to g/cm3 for display
            }
            
            st.success("✅ Calculations completed!")
            
        except Exception as e:
            st.error(f"Error during calculations: {str(e)}")

def display_results():
    """Display the calculation results and visualizations"""
    results = st.session_state.results
    las = st.session_state.las_data
    
    # Metrics summary
    st.subheader("📊 Geomechanical Properties Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Poisson's Ratio</h3>
            <p>Min: <strong>{np.nanmin(results['PR']):.3f}</strong></p>
            <p>Max: <strong>{np.nanmax(results['PR']):.3f}</strong></p>
            <p>Mean: <strong>{np.nanmean(results['PR']):.3f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Young's Modulus</h3>
            <p>Min: <strong>{np.nanmin(results['YM']):.2f} GPa</strong></p>
            <p>Max: <strong>{np.nanmax(results['YM']):.2f} GPa</strong></p>
            <p>Mean: <strong>{np.nanmean(results['YM']):.2f} GPa</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Brittleness Index</h3>
            <p>Vertical Min: <strong>{np.nanmin(results['BRITv']):.3f}</strong></p>
            <p>Vertical Max: <strong>{np.nanmax(results['BRITv']):.3f}</strong></p>
            <p>Horizontal Mean: <strong>{np.nanmean(results['BRITh']):.3f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create interactive plots
    st.subheader("📈 Young's Modulus vs Poisson's Ratio")
    
    # Create figure with subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("YM vs PR (Color by BRITv)", "YM vs PR (Color by BRITh)"),
                        shared_yaxes=True)
    
    # Add scatter plots to subplots
    fig.add_trace(
        go.Scatter(
            x=results['YM'][results['valid_mask']],
            y=results['PR'][results['valid_mask']],
            mode='markers',
            marker=dict(
                color=results['BRITv'][results['valid_mask']],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='BRITv'),
            ),
            name='Data Points',
            text=[f"Depth: {results['depth'][i]:.1f}" for i in range(len(results['depth'])) if results['valid_mask'][i]],
            hovertemplate="YM: %{x:.2f}<br>PR: %{y:.3f}<br>%{text}<extra></extra>"
        ), row=1, col=1)
    
    fig.add_trace(
        go.Scatter(
            x=results['YM'][results['valid_mask']],
            y=results['PR'][results['valid_mask']],
            mode='markers',
            marker=dict(
                color=results['BRITh'][results['valid_mask']],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='BRITh')),
            name='Data Points',
            text=[f"Depth: {results['depth'][i]:.1f}" for i in range(len(results['depth'])) if results['valid_mask'][i]],
            hovertemplate="YM: %{x:.2f}<br>PR: %{y:.3f}<br>%{text}<extra></extra>"
        ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        title_text="Young's Modulus vs Poisson's Ratio with Brittleness Coloring",
        hovermode='closest'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Young's Modulus (GPa)", row=1, col=1)
    fig.update_xaxes(title_text="Young's Modulus (GPa)", row=1, col=2)
    fig.update_yaxes(title_text="Poisson's Ratio", row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Depth plots
    st.subheader("📏 Depth Profiles")
    
    # Create tabs for different properties
    tab1, tab2, tab3, tab4 = st.tabs([
        "Poisson's Ratio", 
        "Young's Modulus", 
        "Anisotropy Parameters", 
        "Brittleness Index"
    ])
    
    with tab1:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=results['PR'],
            y=results['depth'],
            mode='lines',
            name='Poisson Ratio',
            line=dict(color='blue')
        ))
        fig_pr.add_trace(go.Scatter(
            x=results['Vv'],
            y=results['depth'],
            mode='lines',
            name='Vertical PR',
            line=dict(color='green')
        ))
        fig_pr.add_trace(go.Scatter(
            x=results['Vh'],
            y=results['depth'],
            mode='lines',
            name='Horizontal PR',
            line=dict(color='red')
        ))
        fig_pr.update_layout(
            title="Poisson's Ratio vs Depth",
            xaxis_title="Poisson's Ratio",
            yaxis_title="Depth",
            yaxis_autorange='reversed',
            height=600
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    
    with tab2:
        fig_ym = go.Figure()
        fig_ym.add_trace(go.Scatter(
            x=results['YM'],
            y=results['depth'],
            mode='lines',
            name="Young's Modulus",
            line=dict(color='blue')
        ))
        fig_ym.add_trace(go.Scatter(
            x=results['YMv'],
            y=results['depth'],
            mode='lines',
            name='Vertical YM',
            line=dict(color='green')
        ))
        fig_ym.add_trace(go.Scatter(
            x=results['YMh'],
            y=results['depth'],
            mode='lines',
            name='Horizontal YM',
            line=dict(color='red')
        ))
        fig_ym.update_layout(
            title="Young's Modulus vs Depth",
            xaxis_title="Young's Modulus (GPa)",
            yaxis_title="Depth",
            yaxis_autorange='reversed',
            height=600
        )
        st.plotly_chart(fig_ym, use_container_width=True)
    
    with tab3:
        fig_ani = go.Figure()
        fig_ani.add_trace(go.Scatter(
            x=results['delta'],
            y=results['depth'],
            mode='lines',
            name='Delta',
            line=dict(color='blue')
        ))
        fig_ani.add_trace(go.Scatter(
            x=results['epsilon'],
            y=results['depth'],
            mode='lines',
            name='Epsilon',
            line=dict(color='green')
        ))
        fig_ani.add_trace(go.Scatter(
            x=results['gamma'],
            y=results['depth'],
            mode='lines',
            name='Gamma',
            line=dict(color='red')
        ))
        fig_ani.update_layout(
            title="Thomsen Anisotropy Parameters vs Depth",
            xaxis_title="Anisotropy Parameter Value",
            yaxis_title="Depth",
            yaxis_autorange='reversed',
            height=600
        )
        st.plotly_chart(fig_ani, use_container_width=True)
    
    with tab4:
        fig_brit = go.Figure()
        fig_brit.add_trace(go.Scatter(
            x=results['BRITv'],
            y=results['depth'],
            mode='lines',
            name='Vertical Brittleness',
            line=dict(color='blue')
        ))
        fig_brit.add_trace(go.Scatter(
            x=results['BRITh'],
            y=results['depth'],
            mode='lines',
            name='Horizontal Brittleness',
            line=dict(color='green')
        ))
        fig_brit.update_layout(
            title="Brittleness Index vs Depth",
            xaxis_title="Brittleness Index (Normalized)",
            yaxis_title="Depth",
            yaxis_autorange='reversed',
            height=600
        )
        st.plotly_chart(fig_brit, use_container_width=True)
    
    # Download Section
    st.subheader("⬇️ Export Results")
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'DEPTH': results['depth'],
        'VP': results['Vp'],
        'VS': results['Vs'],
        'RHOB': results['Rho'],
        'PR': results['PR'],
        'YM': results['YM'],
        'DELTA': results['delta'],
        'EPSILON': results['epsilon'],
        'GAMMA': results['gamma'],
        'VV': results['Vv'],
        'VH': results['Vh'],
        'YMV': results['YMv'],
        'YMH': results['YMh'],
        'BRITV': results['BRITv'],
        'BRITH': results['BRITh']
    })
    
    # Add original curves from LAS file
    for curve in st.session_state.las_data.keys():
        if curve != 'DEPTH':
            df[curve] = st.session_state.las_data[curve]
    
    # Create CSV buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Create LAS buffer
    las_buffer = BytesIO()
    try:
        new_las = lasio.LASFile()
        new_las.well = las.well
        new_las.header = las.header
        
        # Add original curves
        for curve in las.curves:
            if curve.mnemonic != 'DEPTH':
                new_las.add_curve(
                    curve.mnemonic,
                    las[curve.mnemonic],
                    unit=curve.unit,
                    descr=curve.descr
                )
        
        # Add calculated parameters
        new_las.add_curve('PR', results['PR'], unit="", descr="Poisson's Ratio")
        new_las.add_curve('YM', results['YM'], unit="GPa", descr="Young's Modulus")
        new_las.add_curve('DELTA', results['delta'], unit="", descr="Thomsen Delta")
        new_las.add_curve('EPSILON', results['epsilon'], unit="", descr="Thomsen Epsilon")
        new_las.add_curve('GAMMA', results['gamma'], unit="", descr="Thomsen Gamma")
        new_las.add_curve('VV', results['Vv'], unit="", descr="Vertical Poisson's Ratio")
        new_las.add_curve('VH', results['Vh'], unit="", descr="Horizontal Poisson's Ratio")
        new_las.add_curve('YMV', results['YMv'], unit="GPa", descr="Vertical Young's Modulus")
        new_las.add_curve('YMH', results['YMh'], unit="GPa", descr="Horizontal Young's Modulus")
        new_las.add_curve('BRITV', results['BRITv'], unit="", descr="Vertical Brittleness Index")
        new_las.add_curve('BRITH', results['BRITh'], unit="", descr="Horizontal Brittleness Index")
        
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
            file_name="geomechanical_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📥 Download LAS",
            data=las_buffer.getvalue() if not las_download_disabled else b'',
            file_name="geomechanical_analysis.las",
            mime="application/octet-stream",
            disabled=las_download_disabled,
            use_container_width=True
        )

if __name__ == "__main__":
    main()

