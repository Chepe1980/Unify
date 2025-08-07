import streamlit as st
from PIL import Image
import importlib.util
import sys
from pathlib import Path

# ===========================================
# Page Configuration
# ===========================================
st.set_page_config(
    page_title="GeoAPPS Hub",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================
# Custom CSS Styling
# ===========================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create a style.css file or add these styles directly
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    .logo-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
        padding: 10px;
    }
    .module-title {
        color: #2c3e50;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .module-description {
        color: #7f8c8d;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: white;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# Sidebar with Logo and Navigation
# ===========================================
with st.sidebar:
    # Load your logo image (replace with your actual image path)
    try:
        logo = Image.open("logoApps.JPEGJ")  # Replace with your logo path
        st.image(logo, use_column_width=True, caption="GeoAPPS Hub")
    except:
        st.warning("Logo image not found. Using placeholder.")
        # Placeholder if logo not found
        st.title("GeoAPPS Hub")
    
    st.markdown("---")
    
    # Navigation
    st.subheader("Application Modules")
    app_mode = st.selectbox(
        "Select Module",
        ["Home", "AVAzMOD", "GeoStressMOD", "PasseyTOCMOD", "Machine Learning Algorithms"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    GeoAPPS Hub provides integrated geoscience tools 
    for advanced analysis and modeling.
    """)

# ===========================================
# Main Content Area
# ===========================================
if app_mode == "Home":
    st.title("Welcome to GeoAPPS Hub")
    st.markdown("---")
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Integrated Geoscience Solutions
        Access specialized tools for advanced geoscience analysis, 
        modeling, and machine learning applications.
        """)
        st.markdown("""
        Select a module from the sidebar to get started with your analysis.
        """)
        
    with col2:
        try:
            hero_img = Image.open("hero_image.jpg")  # Replace with your image
            st.image(hero_img, use_column_width=True)
        except:
            st.image("https://via.placeholder.com/400x250", use_column_width=True)
    
    # Module Cards
    st.markdown("## Available Modules")
    st.markdown("---")
    
    cols = st.columns(4)
    modules = [
        ("AVAzMOD", "Azimuthal Velocity Analysis", "📊"),
        ("GeoStressMOD", "Geomechanical Stress Analysis", "⚙️"),
        ("PasseyTOCMOD", "TOC Calculation", "📈"),
        ("Machine Learning", "Advanced ML Algorithms", "🤖")
    ]
    
    for col, (name, desc, icon) in zip(cols, modules):
        with col:
            st.markdown(f"### {icon} {name}")
            st.markdown(f"{desc}")
            if st.button(f"Go to {name}", key=f"btn_{name}"):
                app_mode = name  # This won't work directly - would need session state
    
else:
    # Module loading with improved error handling
    st.markdown(f"# {app_mode}")
    st.markdown("---")
    
    module_map = {
        "AVAzMOD": "AVAzAPP",
        "GeoStressMOD": "GeoMechApp",
        "PasseyTOCMOD": "TocApp_st",
        "Machine Learning Algorithms": "MLalgorithms"
    }
    
    module_name = module_map.get(app_mode)
    
    if module_name:
        try:
            file_path = Path(f"{module_name}.py")
            if not file_path.exists():
                st.error(f"Module file not found: {file_path.resolve()}")
            else:
                with st.spinner(f"Loading {app_mode} module..."):
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    if hasattr(module, 'main'):
                        module.main()
                    else:
                        st.error(f"Module '{module_name}' doesn't have a main() function")
        except Exception as e:
            st.error(f"Error loading {app_mode} module")
            st.exception(e)
    else:
        st.error("Module mapping not found")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2023 GeoAPPS Hub | Version 1.0 | <a href="#">Contact Support</a></p>
</div>
""", unsafe_allow_html=True)



