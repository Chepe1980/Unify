import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

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
    .column-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffdddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
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

def detect_columns(df):
    """Identify required columns with flexible naming"""
    col_map = {
        'depth': ['depth', 'dept', 'md', 'measured depth'],
        'resistivity': ['rt', 'resistivity', 'resist', 'ild'],
        'density': ['rho', 'density', 'den', 'rhob'],
        'porosity': ['phie', 'phi', 'porosity', 'nphi'],
        'youngs_modulus': ['ym', 'youngs modulus', 'youngs_modulus', 'e'],
        'poissons_ratio': ['pr', 'poissons ratio', 'poissons_ratio', 'v']
    }
    
    detected = {}
    for col_type, possible_names in col_map.items():
        for name in possible_names:
            if name.lower() in [c.lower() for c in df.columns]:
                detected[col_type] = name
                break
    
    return detected

def calculate_toc(data, col_map, Ro, Rtbaseline, Rhobaseline):
    """Calculate TOC using Passey method"""
    try:
        Rt = np.array(data[col_map['resistivity']]).astype(float)
        Rho = np.array(data[col_map['density']]).astype(float)
        
        # Calculate cementation exponent (m)
        Phie = data[col_map['porosity']].values if 'porosity' in col_map else None
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
        
        return TOC, DeltaLog, LOM, m
    
    except Exception as e:
        st.error(f"TOC calculation error: {str(e)}")
        return None, None, None, None

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
        st.error(f"Brittleness calculation error: {str(e)}")
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
        st.session_state.results = {
            'TOC': None,
            'DeltaLog': None,
            'LOM': None,
            'm': None,
            'BI': None,
            'brittle_method': None,
            'TOC_corrected': None
        }
    if 'col_map' not in st.session_state:
        st.session_state.col_map = {}
    
    # Data Upload Section
    with st.expander("📤 Upload Data", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            well_file = st.file_uploader("Upload Well Data (Excel)", type=['xlsx', 'xls'])
            if well_file:
                st.session_state.data = load_excel(well_file)
                if st.session_state.data is not None:
                    st.session_state.col_map = detect_columns(st.session_state.data)
                    st.success("Well data loaded successfully!")
        
        with col2:
            toc_file = st.file_uploader("Upload TOC Data (Excel - Optional)", type=['xlsx', 'xls'])
            if toc_file:
                st.session_state.toc_data = load_excel(toc_file)
                if st.session_state.toc_data is not None:
                    st.success("TOC data loaded successfully!")
            
            xrd_file = st.file_uploader("Upload XRD Data (Excel - Optional)", type=['xlsx', 'xls'])
            if xrd_file:
                st.session_state.xrd_data = load_excel(xrd_file)
                if st.session_state.xrd_data is not None:
                    st.success("XRD data loaded successfully!")
    
    # Show detected columns
    if st.session_state.data is not None:
        with st.expander("🔍 Detected Columns", expanded=False):
            st.markdown(f"""
            <div class="column-info">
                <h4>Detected Columns:</h4>
                <ul>
                    <li>Depth: <strong>{st.session_state.col_map.get('depth', 'Not found')}</strong></li>
                    <li>Resistivity: <strong>{st.session_state.col_map.get('resistivity', 'Not found')}</strong></li>
                    <li>Density: <strong>{st.session_state.col_map.get('density', 'Not found')}</strong></li>
                    <li>Porosity: <strong>{st.session_state.col_map.get('porosity', 'Not found (using default m=2.0)')}</strong></li>
                    <li>Young's Modulus: <strong>{st.session_state.col_map.get('youngs_modulus', 'Not found')}</strong></li>
                    <li>Poisson's Ratio: <strong>{st.session_state.col_map.get('poissons_ratio', 'Not found')}</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if 'resistivity' not in st.session_state.col_map or 'density' not in st.session_state.col_map:
                st.markdown(f"""
                <div class="error-box">
                    <h4>⚠️ Missing Required Columns</h4>
                    <p>Please ensure your data contains:</p>
                    <ul>
                        <li>Resistivity column (common names: Rt, Resistivity, Resist, ILD)</li>
                        <li>Density column (common names: Rho, Density, Den, RHOB)</li>
                    </ul>
                    <p>Found columns: {", ".join(st.session_state.data.columns)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.dataframe(st.session_state.data.head())
    
    # TOC Calculation Section
    if (st.session_state.data is not None and 
        'resistivity' in st.session_state.col_map and 
        'density' in st.session_state.col_map):
        
        with st.expander("📈 TOC Calculation", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                Ro = st.number_input("Vitrinite Reflectance (Ro)", 
                                   min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                Rtbaseline = st.number_input("Resistivity Baseline (Rtbaseline)", 
                                           min_value=0.1, value=5.0, step=0.1)
                Rhobaseline = st.number_input("Density Baseline (Rhobaseline)", 
                                            min_value=1.0, max_value=3.0, value=2.65, step=0.01)
                
                if st.button("Calculate TOC", key="calc_toc"):
                    with st.spinner('Calculating TOC...'):
                        TOC, DeltaLog, LOM, m = calculate_toc(
                            st.session_state.data,
                            st.session_state.col_map,
                            Ro, Rtbaseline, Rhobaseline
                        )
                        
                        if TOC is not None:
                            st.session_state.results.update({
                                'TOC': TOC,
                                'DeltaLog': DeltaLog,
                                'LOM': LOM,
                                'm': m
                            })
                            st.success("TOC calculation completed!")
            
            with col2:
                if st.session_state.results.get('LOM') is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Level of Maturity (LOM)</h3>
                        <p>{st.session_state.results['LOM']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Cementation Exponent (m)</h3>
                        <p>{st.session_state.results['m']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Calculate TOC first to see LOM and cementation exponent values")
                
                st.markdown("**TOC Correction**")
                slope = st.number_input("Correction Slope", value=1.0, step=0.1)
                intercept = st.number_input("Correction Intercept", value=0.0, step=0.1)
                
                if st.button("Apply TOC Correction", key="correct_toc"):
                    if st.session_state.results.get('TOC') is not None:
                        corrected_toc = slope * st.session_state.results['TOC'] + intercept
                        st.session_state.results['TOC_corrected'] = corrected_toc
                        st.success("TOC correction applied!")
                    else:
                        st.error("Please calculate TOC first before applying correction")
    
    # Display TOC Results
    if st.session_state.results.get('TOC') is not None:
        with st.expander("📊 TOC Results", expanded=True):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Get depth if available
            depth = (st.session_state.data[st.session_state.col_map['depth']].values 
                    if 'depth' in st.session_state.col_map 
                    else np.arange(len(st.session_state.results['TOC'])))
            
            # TOC Plot
            ax1.plot(st.session_state.results['TOC'], depth, 'k-', label='TOC Passey')
            if st.session_state.results.get('TOC_corrected') is not None:
                ax1.plot(st.session_state.results['TOC_corrected'], depth, 'r-', label='Corrected TOC')
            if st.session_state.toc_data is not None:
                ax1.scatter(st.session_state.toc_data.iloc[:, 1], 
                           st.session_state.toc_data.iloc[:, 0], 
                           c='b', s=40, edgecolor='k', label='TOC RockEval')
            ax1.set_xlabel('TOC (%)')
            ax1.set_ylabel('Depth (m)' if 'depth' in st.session_state.col_map else 'Index')
            ax1.set_title('TOC Profile')
            ax1.grid(True)
            ax1.legend()
            if 'depth' in st.session_state.col_map:
                ax1.invert_yaxis()
            
            # DlogR Plot
            ax2.plot(st.session_state.results['DeltaLog'], depth, 'b-')
            ax2.set_xlabel('ΔlogR')
            ax2.set_ylabel('Depth (m)' if 'depth' in st.session_state.col_map else 'Index')
            ax2.set_title('ΔlogR Profile')
            ax2.grid(True)
            if 'depth' in st.session_state.col_map:
                ax2.invert_yaxis()
            
            st.pyplot(fig)
            
            # Export Results
            with st.expander("💾 Export Results", expanded=False):
                output_df = pd.DataFrame({
                    'Depth': depth,
                    'TOC': st.session_state.results['TOC'],
                    'DeltaLogR': st.session_state.results['DeltaLog'],
                    'LOM': st.session_state.results['LOM'],
                    'Cementation_Exponent': st.session_state.results['m']
                })
                
                if st.session_state.results.get('TOC_corrected') is not None:
                    output_df['TOC_Corrected'] = st.session_state.results['TOC_corrected']
                
                # CSV Export
                csv = output_df.to_csv(index=False).encode()
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="toc_results.csv",
                    mime="text/csv"
                )
                
                # Excel Export
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='TOC_Results')
                    writer.close()
                st.download_button(
                    "Download Excel",
                    data=output.getvalue(),
                    file_name="toc_results.xlsx",
                    mime="application/vnd.ms-excel"
                )
    
    # Brittleness Analysis Section
    if (st.session_state.data is not None and 
        'youngs_modulus' in st.session_state.col_map and 
        'poissons_ratio' in st.session_state.col_map):
        
        with st.expander("💎 Brittleness Analysis", expanded=True):
            method = st.selectbox("Select Method", ["Rickman", "Wang"])
            
            if method == "Rickman":
                if st.button("Calculate Brittleness (Rickman)", key="calc_brittle_rickman"):
                    with st.spinner('Calculating Brittleness Index...'):
                        BI = calculate_brittleness_rickman(
                            st.session_state.data[st.session_state.col_map['youngs_modulus']].values,
                            st.session_state.data[st.session_state.col_map['poissons_ratio']].values
                        )
                        if BI is not None:
                            st.session_state.results['BI'] = BI
                            st.session_state.results['brittle_method'] = "Rickman"
                            st.success("Brittleness calculation completed!")
            
            elif method == "Wang" and st.session_state.xrd_data is not None:
                if st.button("Calculate Brittleness (Wang)", key="calc_brittle_wang"):
                    with st.spinner('Calculating Brittleness Index...'):
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
                            st.error(f"Wang brittleness calculation error: {str(e)}")
            
            if st.session_state.results.get('BI') is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Brittleness Index ({st.session_state.results['brittle_method']} Method)</h3>
                    <p>Range: {np.min(st.session_state.results['BI']):.2f} to {np.max(st.session_state.results['BI']):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 10))
                
                # Get depth if available
                depth = (st.session_state.data[st.session_state.col_map['depth']].values 
                        if 'depth' in st.session_state.col_map 
                        else np.arange(len(st.session_state.results['BI'])))
                
                ax.plot(st.session_state.results['BI'], depth, 'r-')
                ax.set_xlabel('Brittleness Index')
                ax.set_ylabel('Depth (m)' if 'depth' in st.session_state.col_map else 'Index')
                ax.set_title(f'Brittleness Profile ({st.session_state.results["brittle_method"]} Method)')
                ax.grid(True)
                if 'depth' in st.session_state.col_map:
                    ax.invert_yaxis()
                
                st.pyplot(fig)
    
    # Crossplots Section
    if st.session_state.data is not None and st.session_state.col_map:
        with st.expander("📊 Crossplots", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_axis = st.selectbox("X-axis", st.session_state.data.columns)
            
            with col2:
                y_axis = st.selectbox("Y-axis", st.session_state.data.columns)
            
            with col3:
                color_by = st.selectbox("Color by", ["None"] + list(st.session_state.data.columns))
            
            if st.button("Generate Crossplot", key="gen_crossplot"):
                with st.spinner('Generating crossplot...'):
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
                        
                        st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Crossplot generation error: {str(e)}")
            
            if ('youngs_modulus' in st.session_state.col_map and 
                'poissons_ratio' in st.session_state.col_map and 
                st.session_state.results.get('BI') is not None):
                
                if st.button("Generate 3D Plot", key="gen_3dplot"):
                    with st.spinner('Generating 3D plot...'):
                        try:
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            
                            x = st.session_state.data[st.session_state.col_map['poissons_ratio']].values
                            y = st.session_state.data[st.session_state.col_map['youngs_modulus']].values
                            z = st.session_state.results['BI']
                            
                            if st.session_state.results.get('TOC') is not None:
                                sc = ax.scatter(x, y, z, c=st.session_state.results['TOC'], cmap='viridis', s=40)
                                fig.colorbar(sc, ax=ax, label='TOC')
                            else:
                                ax.scatter(x, y, z, s=40)
                            
                            ax.set_xlabel("Poisson's Ratio")
                            ax.set_ylabel("Young's Modulus")
                            ax.set_zlabel("Brittleness Index")
                            ax.set_title("3D Crossplot")
                            
                            st.pyplot(fig)
                        
                        except Exception as e:
                            st.error(f"3D plot generation error: {str(e)}")

    # Help Section
    with st.expander("ℹ️ Help", expanded=False):
        st.markdown("""
        ## PasseyToc - TOC and Brittleness Analysis Tool
        
        ### Features:
        - TOC calculation using Passey's ΔlogR method with LOM correction
        - Brittleness index calculation (Rickman and Wang methods)
        - Interactive visualization of results
        - Data export capabilities
        
        ### Usage Instructions:
        1. **Upload Data**:
           - Well data (required): Should contain resistivity and density columns
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
        
        ### Column Naming:
        The tool automatically detects columns with common names:
        - Resistivity: Rt, Resistivity, Resist, ILD
        - Density: Rho, Density, Den, RHOB
        - Porosity: Phie, Phi, Porosity, NPHI
        - Young's Modulus: YM, Youngs Modulus, Youngs_modulus, E
        - Poisson's Ratio: PR, Poissons Ratio, Poissons_ratio, V
        
        ### Troubleshooting:
        - If you get column detection errors, check your column names
        - For calculation errors, verify your input parameters
        - For unusually high TOC values, check your baseline values
        - Missing values in your data will cause calculation failures
        """)

if __name__ == "__main__":
    main()
