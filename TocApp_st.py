import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
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
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .plot-container {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f9f9f9;
    }
    .metric-card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_excel(file):
    """Load Excel file with proper engine detection"""
    try:
        return pd.read_excel(file, engine='openpyxl')
    except ImportError:
        st.error("Please install openpyxl: pip install openpyxl")
        return None
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def calculate_toc(data, Ro, Rtbaseline, Rhobaseline):
    """Calculate TOC using Passey method"""
    try:
        Rt = data['Rt'].values if 'Rt' in data.columns else None
        Rho = data['Rho'].values if 'Rho' in data.columns else None
        Phie = data['Phie'].values if 'Phie' in data.columns else None
        
        if Rt is None or Rho is None:
            st.error("Resistivity (Rt) and Density (Rho) data required")
            return None, None, None
        
        # Calculate cementation exponent (m)
        m = 1.20 + 12.76 * Phie if Phie is not None else 2.0
        
        # Calculate LOM
        exp_term = np.exp(1 - 2.778 * Ro)
        denominator = 0.59 + 0.41 * (exp_term ** 28.45)
        LOM = 8.18 * (Ro / denominator) ** (1 / m)
        
        # Calculate DeltaLogR
        DeltaLog = np.log10(Rt / Rtbaseline) + 2.5 * (Rho - Rhobaseline)
        
        # Calculate TOC
        a = 0.0297 - 0.1688 * LOM
        TOC = DeltaLog * 10 * np.exp(a)
        
        return TOC, DeltaLog, LOM
    
    except Exception as e:
        st.error(f"Error calculating TOC: {str(e)}")
        return None, None, None

def calculate_brittleness_rickman(YM, PR):
    """Calculate brittleness using Rickman method"""
    try:
        # Get min/max values
        Emin, Emax = np.min(YM), np.max(YM)
        vmin, vmax = np.min(PR), np.max(PR)
        
        # Normalize
        En = 100 * (YM - Emin) / (Emax - Emin)
        vn = 100 * (PR - vmax) / (vmin - vmax)
        
        # Calculate brittleness
        BI = (En + vn) / 2
        
        return BI
    
    except Exception as e:
        st.error(f"Error calculating brittleness: {str(e)}")
        return None

def main():
    st.title("⛏️ PasseyToc - TOC and Brittleness Analysis")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'toc_data' not in st.session_state:
        st.session_state.toc_data = None
    if 'xrd_data' not in st.session_state:
        st.session_state.xrd_data = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Data Upload Section
    with st.expander("📤 Upload Data", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            well_file = st.file_uploader("Upload Well Data (Excel)", type=['xlsx', 'xls'])
            if well_file:
                st.session_state.data = load_excel(well_file)
                if st.session_state.data is not None:
                    st.success("Well data loaded successfully!")
                    st.dataframe(st.session_state.data.head())
        
        with col2:
            toc_file = st.file_uploader("Upload TOC Data (Excel - Optional)", type=['xlsx', 'xls'])
            if toc_file:
                st.session_state.toc_data = load_excel(toc_file)
                if st.session_state.toc_data is not None:
                    st.success("TOC data loaded successfully!")
                    st.dataframe(st.session_state.toc_data.head())
            
            xrd_file = st.file_uploader("Upload XRD Data (Excel - Optional)", type=['xlsx', 'xls'])
            if xrd_file:
                st.session_state.xrd_data = load_excel(xrd_file)
                if st.session_state.xrd_data is not None:
                    st.success("XRD data loaded successfully!")
                    st.dataframe(st.session_state.xrd_data.head())
    
    # TOC Calculation Section
    if st.session_state.data is not None:
        with st.expander("📈 TOC Calculation", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                Ro = st.number_input("Vitrinite Reflectance (Ro)", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
                Rtbaseline = st.number_input("Resistivity Baseline (Rtbaseline)", value=5.0, min_value=0.1, step=0.1)
                Rhobaseline = st.number_input("Density Baseline (Rhobaseline)", value=2.65, min_value=1.0, max_value=3.0, step=0.01)
                
                if st.button("Calculate TOC", key="calc_toc"):
                    TOC, DeltaLog, LOM = calculate_toc(st.session_state.data, Ro, Rtbaseline, Rhobaseline)
                    if TOC is not None:
                        st.session_state.results.update({
                            'TOC': TOC,
                            'DeltaLog': DeltaLog,
                            'LOM': LOM
                        })
                        st.success("TOC calculation completed!")
            
            with col2:
                if 'TOC' in st.session_state.results:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Level of Maturity (LOM)</h3>
                        <p>{st.session_state.results['LOM']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TOC Correction
                    slope = st.number_input("Correction Slope", value=1.0, step=0.1)
                    intercept = st.number_input("Correction Intercept", value=0.0, step=0.1)
                    
                    if st.button("Apply TOC Correction", key="correct_toc"):
                        corrected_toc = slope * st.session_state.results['TOC'] + intercept
                        st.session_state.results['TOC_corrected'] = corrected_toc
                        st.success("TOC correction applied!")
    
    # Display TOC Results
    if 'TOC' in st.session_state.results:
        with st.expander("📊 TOC Results", expanded=True):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # TOC Plot
            ax1.plot(st.session_state.results['TOC'], st.session_state.data['DEPTH'], 'k-', label='TOC Passey')
            if 'TOC_corrected' in st.session_state.results:
                ax1.plot(st.session_state.results['TOC_corrected'], st.session_state.data['DEPTH'], 'r-', label='Corrected TOC')
            if st.session_state.toc_data is not None:
                ax1.scatter(st.session_state.toc_data.iloc[:, 1], st.session_state.toc_data.iloc[:, 0], 
                           c='b', s=40, edgecolor='k', label='TOC RockEval')
            ax1.set_xlabel('TOC')
            ax1.set_ylabel('Depth (m)')
            ax1.set_title('TOC Profile')
            ax1.grid(True)
            ax1.invert_yaxis()
            ax1.legend()
            
            # DlogR Plot
            ax2.plot(st.session_state.results['DeltaLog'], st.session_state.data['DEPTH'], 'b-')
            ax2.set_xlabel('DlogR')
            ax2.set_ylabel('Depth (m)')
            ax2.set_title('DlogR Profile')
            ax2.grid(True)
            ax2.invert_yaxis()
            
            st.pyplot(fig)
    
    # Brittleness Analysis Section
    if st.session_state.data is not None and ('YM' in st.session_state.data.columns and 'PR' in st.session_state.data.columns):
        with st.expander("💎 Brittleness Analysis", expanded=True):
            method = st.selectbox("Select Method", ["Rickman", "Wang"])
            
            if method == "Rickman":
                if st.button("Calculate Brittleness (Rickman)", key="calc_brittle_rickman"):
                    BI = calculate_brittleness_rickman(
                        st.session_state.data['YM'].values,
                        st.session_state.data['PR'].values
                    )
                    if BI is not None:
                        st.session_state.results['BI'] = BI
                        st.session_state.results['brittle_method'] = "Rickman"
                        st.success("Brittleness calculation completed!")
            
            elif method == "Wang" and st.session_state.xrd_data is not None:
                if st.button("Calculate Brittleness (Wang)", key="calc_brittle_wang"):
                    try:
                        Qz = st.session_state.xrd_data.iloc[:, 1].values
                        Dlm = st.session_state.xrd_data.iloc[:, 2].values
                        CLc = st.session_state.xrd_data.iloc[:, 3].values
                        Cly = st.session_state.xrd_data.iloc[:, 4].values
                        ToC1 = st.session_state.xrd_data.iloc[:, 5].values / 100
                        
                        BrittleWang = (Qz + Dlm) / (Qz + Dlm + CLc + Cly + ToC1) * 100
                        st.session_state.results['BI'] = BrittleWang
                        st.session_state.results['brittle_method'] = "Wang"
                        st.success("Brittleness calculation completed!")
                    except Exception as e:
                        st.error(f"Error calculating Wang brittleness: {str(e)}")
            
            if 'BI' in st.session_state.results:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Brittleness Index ({st.session_state.results['brittle_method']} Method)</h3>
                    <p>Range: {np.min(st.session_state.results['BI']):.2f} to {np.max(st.session_state.results['BI']):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 10))
                ax.plot(st.session_state.results['BI'], st.session_state.data['DEPTH'], 'r-')
                ax.set_xlabel('Brittleness Index')
                ax.set_ylabel('Depth (m)')
                ax.set_title(f'Brittleness Profile ({st.session_state.results["brittle_method"]} Method)')
                ax.grid(True)
                ax.invert_yaxis()
                st.pyplot(fig)
    
    # Crossplots Section
    if st.session_state.data is not None and 'available_curves' in st.session_state:
        with st.expander("📊 Crossplots", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_axis = st.selectbox("X-axis", st.session_state.available_curves)
            
            with col2:
                y_axis = st.selectbox("Y-axis", st.session_state.available_curves)
            
            with col3:
                color_by = st.selectbox("Color by", ["None"] + st.session_state.available_curves)
            
            if st.button("Generate Crossplot", key="gen_crossplot"):
                try:
                    x = st.session_state.data[x_axis].values
                    y = st.session_state.data[y_axis].values
                    c = st.session_state.data[color_by].values if color_by != "None" else None
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    if c is not None:
                        sc = ax.scatter(x, y, c=c, cmap='viridis', s=30, edgecolor='k')
                        plt.colorbar(sc, ax=ax, label=color_by)
                    else:
                        ax.scatter(x, y, s=30, edgecolor='k')
                    
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    ax.set_title(f"{y_axis} vs {x_axis}")
                    ax.grid(True)
                    
                    st.session_state.results['crossplot'] = fig
                    st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error generating crossplot: {str(e)}")
            
            if 'YM' in st.session_state.data.columns and 'PR' in st.session_state.data.columns and 'BI' in st.session_state.results:
                if st.button("Generate 3D Plot", key="gen_3dplot"):
                    try:
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        if 'TOC' in st.session_state.results:
                            sc = ax.scatter(
                                st.session_state.data['PR'],
                                st.session_state.data['YM'],
                                st.session_state.results['BI'],
                                c=st.session_state.results['TOC'],
                                cmap='viridis',
                                s=40
                            )
                            fig.colorbar(sc, ax=ax, label='TOC')
                        else:
                            ax.scatter(
                                st.session_state.data['PR'],
                                st.session_state.data['YM'],
                                st.session_state.results['BI'],
                                s=40
                            )
                        
                        ax.set_xlabel("Poisson's Ratio")
                        ax.set_ylabel("Young's Modulus")
                        ax.set_zlabel("Brittleness Index")
                        ax.set_title("3D Crossplot")
                        
                        st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error generating 3D plot: {str(e)}")

    # Help Section
    with st.expander("ℹ️ Help", expanded=False):
        st.markdown("""
        ## PasseyToc - TOC and Brittleness Analysis Tool
        
        ### Features:
        - TOC calculation using Passey's ΔlogR method with LOM correction
        - Brittleness index calculation (Rickman and Wang methods)
        - Interactive visualization of results
        
        ### Usage Instructions:
        1. **Upload Data**:
           - Well data (required): Should contain at least Depth, Rt (resistivity), and Rho (density) columns
           - TOC data (optional): For validation of calculated TOC
           - XRD data (optional): Required for Wang brittleness method
        
        2. **TOC Calculation**:
           - Enter Ro, Rtbaseline, and Rhobaseline values
           - Click "Calculate TOC" button
           - Optionally apply correction factors
        
        3. **Brittleness Analysis**:
           - Select calculation method (Rickman or Wang)
           - Click corresponding calculation button
        
        4. **Visualization**:
           - Explore results through interactive plots
           - Generate custom crossplots
        """)

if __name__ == "__main__":
    main()
