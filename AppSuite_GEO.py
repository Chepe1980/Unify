echAppimport streamlit as st
from PIL import Image
import importlib.util
import sys
from pathlib import Path

# ===========================================
# PAGE CONFIGURATION
# ===========================================
st.set_page_config(
    page_title="GeoAPPS Hub",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================
# CUSTOM CSS STYLING
# ===========================================
def inject_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f5f5f5;
            padding: 2rem;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2c3e50 0%, #1a2530 100%);
            color: white;
        }
        
        /* Logo styling */
        .logo-container {
            padding: 1rem;
            text-align: center;
            border-bottom: 1px solid #444;
        }
        
        /* Module cards */
        .module-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            transition: transform 0.2s;
        }
        
        .module-card:hover {
            transform: translateY(-5px);
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        /* Titles */
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ===========================================
# SIDEBAR LAYOUT
# ===========================================
with st.sidebar:
    # Logo section
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    try:
        logo = Image.open("logoApps.png")
        st.image(logo, use_column_width=True)
    except:
        st.title("GeoAPPS Hub")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    st.markdown("## Navigation")
    app_mode = st.selectbox(
        "Select Module",
        ["Home", "AVAzMOD", "GeoStressMOD", "PasseyTOCMOD", "Machine Learning"],
        label_visibility="collapsed"
    )
    
    # About section
    st.markdown("---")
    st.markdown("""
    **About GeoAPPS Hub**  
    Integrated geoscience tools for advanced analysis  
    Version 1.0.0  
    [Contact Support](mailto:alfredoguerrero1980@gmail.com)
    """)

# ===========================================
# MODULE LOADING FUNCTION
# ===========================================
def load_module(module_name, function_name="main"):
    """Dynamically loads a module and executes its main function"""
    try:
        file_path = Path(f"modules/{module_name}.py")
        
        if not file_path.exists():
            st.error(f"Module file not found: {file_path}")
            return False
            
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, function_name):
            st.error(f"Module '{module_name}' is missing required function '{function_name}()'")
            available = [f for f in dir(module) if not f.startswith('__')]
            st.info(f"Available functions: {', '.join(available)}")
            return False
            
        # Execute the module's main function
        getattr(module, function_name)()
        return True
        
    except Exception as e:
        st.error(f"Error loading {module_name} module")
        st.exception(e)
        return False

# ===========================================
# MAIN CONTENT AREA
# ===========================================
if app_mode == "Home":
    st.title("Welcome to GeoAPPS Hub developed By PhD. J.A. Guerrero Castro")
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Integrated Geoscience Solutions
        Access specialized tools for advanced geoscience analysis, 
        modeling, and machine learning applications.
        """)
        
    with col2:
        try:
            hero_img = Image.open("assets/hero_image.jpg")
            st.image(hero_img, use_column_width=True)
        except:
            st.image("https://via.placeholder.com/500x300?text=GeoAPP+Hero+Image", 
                    use_column_width=True)
    
    # Module showcase
    st.markdown("## Available Modules")
    st.markdown("---")
    
    modules = [
        {
            "name": "AVAzMOD",
            "icon": "🧭",
            "description": "Azimuthal Velocity Analysis for fracture characterization and Fluid Substitution in shale rocks",
            "color": "#3498db"
        },
        {
            "name": "GeoStressMOD",
            "icon": "⚙️",
            "description": "Geomechanical stress analysis and modeling of Hoop Stress wellbore",
            "color": "#e74c3c"
        },
        {
            "name": "PasseyTOCMOD",
            "icon": "📊",
            "description": "Total Organic Carbon calculation using ΔLogR method of Passey´s method",
            "color": "#2ecc71"
        },
        {
            "name": "Machine Learning",
            "icon": "🤖",
            "description": "Advanced ML algorithms for geoscience applications Sonic Log Prediction",
            "color": "#9b59b6"
        }
    ]
    
    cols = st.columns(4)
    for idx, module in enumerate(modules):
        with cols[idx]:
            st.markdown(f"""
            <div class="module-card" style="border-top: 4px solid {module['color']}">
                <h3>{module['icon']} {module['name']}</h3>
                <p>{module['description']}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Module loading logic
    st.title(f"{app_mode}")
    st.markdown("---")
    
    module_mapping = {
        "AVAzMOD": ("AVAzAPP", "main"),
        "GeoStressMOD": ("GeoMechApp", "main"),
        "PasseyTOCMOD": ("TocApp_st", "main"),
        "Machine Learning": ("MLalgorithms", "main")
    }
    
    if app_mode in module_mapping:
        module_info = module_mapping[app_mode]
        with st.spinner(f"Loading {app_mode} module..."):
            if not load_module(module_info[0], module_info[1]):
                st.warning(f"Failed to load {app_mode} module. Check the error messages above.")
    else:
        st.error("Module configuration error: Unknown module selected")

# ===========================================
# FOOTER
# ===========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #7f8c8d; font-size: 0.9rem;">
    © 2023 GeoAPPS Hub | Developed by Geoscience Team | v1.0.0
</div>
""", unsafe_allow_html=True)







