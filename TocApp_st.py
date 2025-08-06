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

def load_excel(file):
    """Load Excel file with proper engine detection"""
    try:
        # First try with openpyxl
        return pd.read_excel(file, engine='openpyxl')
    except ImportError:
        try:
            # Fallback to xlrd
            return pd.read_excel(file, engine='xlrd')
        except ImportError:
            st.error("""
            Missing required Excel reader. Please install one of:
            - pip install openpyxl (recommended)
            - pip install xlrd
            """)
            return None
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

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
                st.session_state.data = load_excel(well_file)
                if st.session_state.data is not None:
                    st.success("Well data loaded successfully!")
                    
                    # Display preview
                    st.dataframe(st.session_state.data.head())
                    
                    # Show available curves
                    available_curves = [col for col in st.session_state.data.columns if col != 'DEPTH']
                    st.session_state.available_curves = available_curves
                    st.write("Available curves:", ", ".join(available_curves))
        
        with col2:
            st.subheader("TOC Data")
            toc_file = st.file_uploader("Upload TOC Data (Excel)", type=['xlsx', 'xls'])
            if toc_file:
                st.session_state.toc_data = load_excel(toc_file)
                if st.session_state.toc_data is not None:
                    st.success("TOC data loaded successfully!")
                    st.dataframe(st.session_state.toc_data.head())
            
            st.subheader("XRD Data")
            xrd_file = st.file_uploader("Upload XRD Data (Excel)", type=['xlsx', 'xls'])
            if xrd_file:
                st.session_state.xrd_data = load_excel(xrd_file)
                if st.session_state.xrd_data is not None:
                    st.success("XRD data loaded successfully!")
                    st.dataframe(st.session_state.xrd_data.head())

    # [Rest of your code remains the same...]
    # Include all the other functions (calculate_toc, apply_toc_correction, etc.)
    # from the previous implementation

if __name__ == "__main__":
    main()
