import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import seaborn as sns
from ternary.ternary_axes_subplot import TernaryAxesSubplot
from mpl_toolkits.mplot3d import Axes3D

# Configure the app
st.set_page_config(
    page_title="PasseyToc - TOC and Brittleness Analysis",
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
    .tab-content {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("⛏️ PasseyToc - TOC and Brittleness Analysis")
    st.markdown("""
    This app implements the Passey ΔlogR method for Total Organic Carbon (TOC) estimation 
    from well log data, along with brittleness index calculations.
    """)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'toc_data' not in st.session_state:
        st.session_state.toc_data = None
    if 'xrd_data' not in st.session_state:
        st.session_state.xrd_data = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Input Data", "TOC Analysis", "Brittleness", "Crossplots", "Help"])
    
    with tab1:
        st.header("📊 Data Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Well Data")
            well_file = st.file_uploader("Upload Well Data (Excel)", type=['xlsx', 'xls'])
            if well_file:
                try:
                    st.session_state.data = pd.read_excel(well_file)
                    st.success("Well data loaded successfully!")
                    
                    # Display preview
                    st.dataframe(st.session_state.data.head())
                    
                    # Show available curves
                    available_curves = [col for col in st.session_state.data.columns if col != 'DEPTH']
                    st.session_state.available_curves = available_curves
                    st.write("Available curves:", ", ".join(available_curves))
                    
                except Exception as e:
                    st.error(f"Error loading well data: {str(e)}")
        
        with col2:
            st.subheader("TOC Data")
            toc_file = st.file_uploader("Upload TOC Data (Excel)", type=['xlsx', 'xls'])
            if toc_file:
                try:
                    st.session_state.toc_data = pd.read_excel(toc_file)
                    st.success("TOC data loaded successfully!")
                    st.dataframe(st.session_state.toc_data.head())
                except Exception as e:
                    st.error(f"Error loading TOC data: {str(e)}")
            
            st.subheader("XRD Data")
            xrd_file = st.file_uploader("Upload XRD Data (Excel)", type=['xlsx', 'xls'])
            if xrd_file:
                try:
                    st.session_state.xrd_data = pd.read_excel(xrd_file)
                    st.success("XRD data loaded successfully!")
                    st.dataframe(st.session_state.xrd_data.head())
                except Exception as e:
                    st.error(f"Error loading XRD data: {str(e)}")
    
    with tab2:
        st.header("📈 TOC Analysis")
        
        if st.session_state.data is None:
            st.warning("Please upload well data first")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Parameters")
                st.session_state.Ro = st.number_input("Vitrinite Reflectance (Ro)", value=0.5)
                st.session_state.Rtbaseline = st.number_input("Resistivity Baseline (Rtbaseline)", value=5.0)
                st.session_state.Rhobaseline = st.number_input("Density Baseline (Rhobaseline)", value=2.65)
                
                if st.button("Calculate TOC"):
                    calculate_toc()
            
            with col2:
                st.subheader("TOC Correction")
                slope = st.number_input("Slope", value=1.0)
                intercept = st.number_input("Intercept", value=0.0)
                
                if st.button("Apply TOC Correction"):
                    apply_toc_correction(slope, intercept)
            
            if 'TOC' in st.session_state.results:
                st.subheader("Results")
                
                fig, ax = plt.subplots(figsize=(8, 10))
                ax.plot(st.session_state.results['TOC'], st.session_state.data['DEPTH'], 'k-', label='TOC Passey')
                
                if 'TOC_corrected' in st.session_state.results:
                    ax.plot(st.session_state.results['TOC_corrected'], st.session_state.data['DEPTH'], 'r-', label='Corrected TOC')
                
                if st.session_state.toc_data is not None:
                    ax.scatter(st.session_state.toc_data.iloc[:, 1], st.session_state.toc_data.iloc[:, 0], 
                               c='b', s=40, edgecolor='k', label='TOC RockEval')
                
                ax.set_xlabel('TOC')
                ax.set_ylabel('Depth (m)')
                ax.set_title('TOC Profile')
                ax.grid(True)
                ax.invert_yaxis()
                ax.legend()
                st.pyplot(fig)
                
                # Show DlogR plot
                if 'DeltaLog' in st.session_state.results:
                    fig2, ax2 = plt.subplots(figsize=(8, 10))
                    ax2.plot(st.session_state.results['DeltaLog'], st.session_state.data['DEPTH'], 'k-')
                    ax2.set_xlabel('DlogR')
                    ax2.set_ylabel('Depth (m)')
                    ax2.set_title('DlogR Profile')
                    ax2.grid(True)
                    ax2.invert_yaxis()
                    st.pyplot(fig2)
    
    with tab3:
        st.header("💎 Brittleness Analysis")
        
        if st.session_state.data is None:
            st.warning("Please upload well data first")
        else:
            method = st.selectbox("Brittleness Method", ["Rickman", "Wang"])
            
            if method == "Rickman":
                if 'YM' not in st.session_state.data.columns or 'PR' not in st.session_state.data.columns:
                    st.warning("Young's Modulus and Poisson's Ratio data required")
                else:
                    if st.button("Calculate Brittleness (Rickman)"):
                        calculate_brittleness_rickman()
            
            elif method == "Wang":
                if st.session_state.xrd_data is None:
                    st.warning("XRD data required for Wang method")
                else:
                    if st.button("Calculate Brittleness (Wang)"):
                        calculate_brittleness_wang()
            
            if 'BI' in st.session_state.results:
                st.subheader("Brittleness Results")
                
                fig, ax = plt.subplots(figsize=(8, 10))
                ax.plot(st.session_state.results['BI'], st.session_state.data['DEPTH'], 'r-')
                ax.set_xlabel('Brittleness Index')
                ax.set_ylabel('Depth (m)')
                ax.set_title('Brittleness Profile')
                ax.grid(True)
                ax.invert_yaxis()
                st.pyplot(fig)
                
                # Show Young's Modulus and Poisson's Ratio if available
                if 'YM' in st.session_state.data.columns and 'PR' in st.session_state.data.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_ym, ax_ym = plt.subplots(figsize=(6, 8))
                        ax_ym.plot(st.session_state.data['YM'], st.session_state.data['DEPTH'], 'b-')
                        ax_ym.set_xlabel("Young's Modulus")
                        ax_ym.set_ylabel('Depth (m)')
                        ax_ym.grid(True)
                        ax_ym.invert_yaxis()
                        st.pyplot(fig_ym)
                    
                    with col2:
                        fig_pr, ax_pr = plt.subplots(figsize=(6, 8))
                        ax_pr.plot(st.session_state.data['PR'], st.session_state.data['DEPTH'], 'g-')
                        ax_pr.set_xlabel("Poisson's Ratio")
                        ax_pr.set_ylabel('Depth (m)')
                        ax_pr.grid(True)
                        ax_pr.invert_yaxis()
                        st.pyplot(fig_pr)
    
    with tab4:
        st.header("📊 Crossplots")
        
        if st.session_state.data is None:
            st.warning("Please upload well data first")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_axis = st.selectbox("X-axis", st.session_state.available_curves)
            
            with col2:
                y_axis = st.selectbox("Y-axis", st.session_state.available_curves)
            
            with col3:
                color_by = st.selectbox("Color by", ["None"] + st.session_state.available_curves)
            
            if st.button("Generate Crossplot"):
                generate_crossplot(x_axis, y_axis, color_by)
            
            if 'crossplot' in st.session_state.results:
                st.pyplot(st.session_state.results['crossplot'])
            
            if st.button("Generate 3D Plot"):
                generate_3d_plot()
    
    with tab5:
        st.header("ℹ️ Help")
        
        st.markdown("""
        ## PasseyToc - TOC and Brittleness Analysis Tool
        
        This tool implements the Passey ΔlogR method for Total Organic Carbon (TOC) estimation 
        from well log data, along with brittleness index calculations.
        
        ### Features:
        - TOC calculation using Passey's method with LOM correction
        - Brittleness index calculation (Rickman and Wang methods)
        - Dynamic to static Young's Modulus conversion
        - Crossplot visualization capabilities
        - Ternary plots for mineralogy analysis
        
        ### Usage:
        1. Load your well data (Excel format)
        2. Enter required parameters (Ro, baselines)
        3. Calculate LOM and TOC
        4. Explore results through various plots
        
        ### Data Requirements:
        - Well data should include at least resistivity (Rt) and density (Rho) logs
        - For TOC validation, upload TOC measurements (optional)
        - For XRD-based brittleness, upload XRD mineralogy data
        """)

def calculate_toc():
    """Calculate TOC using Passey method"""
    try:
        data = st.session_state.data
        Rt = data['Rt'].values if 'Rt' in data.columns else None
        Rho = data['Rho'].values if 'Rho' in data.columns else None
        Phie = data['Phie'].values if 'Phie' in data.columns else None
        
        if Rt is None or Rho is None:
            st.error("Resistivity (Rt) and Density (Rho) data required")
            return
        
        # Calculate cementation exponent (m)
        if Phie is not None:
            m = 1.20 + 12.76 * Phie  # Case 1 Porosity > 0.10
        else:
            m = 2.0  # Default value if porosity not available
        
        # Calculate LOM
        Ro = st.session_state.Ro
        exp_term = np.exp(1 - 2.778 * Ro)
        denominator = 0.59 + 0.41 * (exp_term ** 28.45)
        LOM1 = 8.18 * (Ro / denominator) ** (1 / m)
        LOM2 = np.max(LOM1)
        
        # Calculate DeltaLogR
        Rtbaseline = st.session_state.Rtbaseline
        Rhobaseline = st.session_state.Rhobaseline
        DeltaLog = np.log10(Rt / Rtbaseline) + 2.5 * (Rho - Rhobaseline)
        
        # Calculate TOC
        a = 0.0297 - 0.1688 * LOM2
        TOC = DeltaLog * 10 * np.exp(a)
        
        # Store results
        st.session_state.results = {
            'TOC': TOC,
            'DeltaLog': DeltaLog,
            'LOM': LOM2,
            'm': m
        }
        
        st.success("TOC calculation completed!")
    
    except Exception as e:
        st.error(f"Error calculating TOC: {str(e)}")

def apply_toc_correction(slope, intercept):
    """Apply linear correction to TOC values"""
    try:
        if 'TOC' not in st.session_state.results:
            st.warning("Calculate TOC first")
            return
        
        TOC_corrected = slope * st.session_state.results['TOC'] + intercept
        st.session_state.results['TOC_corrected'] = TOC_corrected
        st.success("TOC correction applied!")
    
    except Exception as e:
        st.error(f"Error applying TOC correction: {str(e)}")

def calculate_brittleness_rickman():
    """Calculate brittleness using Rickman method"""
    try:
        data = st.session_state.data
        YM = data['YM'].values if 'YM' in data.columns else None
        PR = data['PR'].values if 'PR' in data.columns else None
        
        if YM is None or PR is None:
            st.error("Young's Modulus (YM) and Poisson's Ratio (PR) data required")
            return
        
        # Get min/max values
        Emin, Emax = np.min(YM), np.max(YM)
        vmin, vmax = np.min(PR), np.max(PR)
        
        # Normalize
        En = 100 * (YM - Emin) / (Emax - Emin)
        vn = 100 * (PR - vmax) / (vmin - vmax)
        
        # Calculate brittleness
        BI = (En + vn) / 2
        
        # Store results
        st.session_state.results['BI'] = BI
        st.session_state.results['brittle_method'] = "Rickman"
        
        st.success("Brittleness (Rickman) calculation completed!")
    
    except Exception as e:
        st.error(f"Error calculating brittleness: {str(e)}")

def calculate_brittleness_wang():
    """Calculate brittleness using Wang method"""
    try:
        if st.session_state.xrd_data is None:
            st.error("XRD data required")
            return
        
        xrd_data = st.session_state.xrd_data
        Qz = xrd_data.iloc[:, 1].values if xrd_data.shape[1] > 1 else None
        Dlm = xrd_data.iloc[:, 2].values if xrd_data.shape[1] > 2 else None
        CLc = xrd_data.iloc[:, 3].values if xrd_data.shape[1] > 3 else None
        Cly = xrd_data.iloc[:, 4].values if xrd_data.shape[1] > 4 else None
        ToC1 = xrd_data.iloc[:, 5].values / 100 if xrd_data.shape[1] > 5 else None
        
        if None in [Qz, Dlm, CLc, Cly, ToC1]:
            st.error("XRD data should include Qz, Dlm, CLc, Cly, and ToC1 columns")
            return
        
        BrittleWang = (Qz + Dlm) / (Qz + Dlm + CLc + Cly + ToC1) * 100
        
        # Store results
        st.session_state.results['BI'] = BrittleWang
        st.session_state.results['brittle_method'] = "Wang"
        
        st.success("Brittleness (Wang) calculation completed!")
    
    except Exception as e:
        st.error(f"Error calculating brittleness: {str(e)}")

def generate_crossplot(x_axis, y_axis, color_by):
    """Generate 2D crossplot"""
    try:
        data = st.session_state.data
        
        x = data[x_axis].values if x_axis in data.columns else None
        y = data[y_axis].values if y_axis in data.columns else None
        
        if x is None or y is None:
            st.error("Selected axis data not found")
            return
        
        if color_by != "None":
            c = data[color_by].values if color_by in data.columns else None
        else:
            c = None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if c is not None:
            sc = ax.scatter(x, y, c=c, cmap='jet', s=30, edgecolor='k')
            plt.colorbar(sc, ax=ax, label=color_by)
        else:
            ax.scatter(x, y, s=30, edgecolor='k')
        
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{y_axis} vs {x_axis}")
        ax.grid(True)
        
        st.session_state.results['crossplot'] = fig
        st.success("Crossplot generated!")
    
    except Exception as e:
        st.error(f"Error generating crossplot: {str(e)}")

def generate_3d_plot():
    """Generate 3D crossplot"""
    try:
        data = st.session_state.data
        
        if 'PR' not in data.columns or 'YM' not in data.columns or 'BI' not in st.session_state.results:
            st.error("Poisson's Ratio, Young's Modulus and Brittleness data required")
            return
        
        PR = data['PR'].values
        YM = data['YM'].values
        BI = st.session_state.results['BI']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if 'TOC' in st.session_state.results:
            sc = ax.scatter(PR, YM, BI, c=st.session_state.results['TOC'], cmap='jet', s=40)
            fig.colorbar(sc, ax=ax, label='TOC')
        else:
            ax.scatter(PR, YM, BI, s=40)
        
        ax.set_xlabel("Poisson's Ratio")
        ax.set_ylabel("Young's Modulus")
        ax.set_zlabel("Brittleness")
        ax.set_title("3D Crossplot")
        
        st.session_state.results['3dplot'] = fig
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error generating 3D plot: {str(e)}")

if __name__ == "__main__":
    main()
