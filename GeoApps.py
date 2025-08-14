import streamlit as st
from PIL import Image
import importlib.util
import sys
from pathlib import Path
import os

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
        
        /* Error messages */
        .stAlert {
            border-left: 4px solid #e74c3c;
        }
        
        /* Developer photo styling */
        .developer-photo {
            border-radius: 30%;
            width: 100px;
            height:200px;
            object-fit: cover;
            border: 3px solid #3498db;
            margin: 0 auto;
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ===========================================
# MODULE LOADING SYSTEM (IMPROVED)
# ===========================================
def create_placeholder_module(module_path, module_name):
    """Creates a placeholder module if it doesn't exist"""
    placeholder_code = f'''# {module_name} placeholder
import streamlit as st

def main():
    """Placeholder module for {module_name}"""
    st.title("{module_name.replace('_', ' ').title()} Module")
    st.image("https://via.placeholder.com/800x400?text={module_name}+Module", 
             use_column_width=True)
    st.warning("This is a placeholder module")
    st.info("The actual implementation will go here")
    
    if st.button("Sample Button"):
        st.success("Button clicked!")

    st.markdown("### Sample DataFrame")
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'])
    st.dataframe(df)
'''
    with open(module_path, 'w') as f:
        f.write(placeholder_code)

def load_module(module_name, function_name="main"):
    """Improved module loader with better error handling and placeholder creation"""
    try:
        # Try multiple possible locations
        possible_locations = [
            Path("modules"),  # ./modules/
            Path("."),        # current directory
            Path("src/modules")  # ./src/modules/
        ]
        
        module_path = None
        for location in possible_locations:
            potential_path = location / f"{module_name}.py"
            if potential_path.exists():
                module_path = potential_path
                break
        
        # If not found, create a placeholder in the modules directory
        if not module_path:
            modules_dir = Path("modules")
            modules_dir.mkdir(exist_ok=True)
            module_path = modules_dir / f"{module_name}.py"
            create_placeholder_module(module_path, module_name)
            st.warning(f"Created placeholder module at {module_path}")
        
        # Debug information
        debug_expander = st.expander("Debug Information", expanded=False)
        with debug_expander:
            st.write(f"Loading module from: {module_path}")
            st.write("Current working directory:", os.getcwd())
            st.write("Directory contents:", os.listdir())
            if Path("modules").exists():
                st.write("Modules directory contents:", os.listdir("modules"))
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, function_name):
            available_functions = [f for f in dir(module) if not f.startswith('_')]
            raise AttributeError(
                f"Module '{module_name}' is missing required function '{function_name}()'\n"
                f"Available functions: {', '.join(available_functions)}"
            )
        
        # Execute the module's main function
        getattr(module, function_name)()
        return True
        
    except Exception as e:
        st.error(f"Error loading {module_name} module")
        st.exception(e)
        return False

# ===========================================
# SIDEBAR LAYOUT
# ===========================================
with st.sidebar:
    # Logo section
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    try:
        logo = Image.open("logoApps.png")
        st.image(logo, use_container_width=True)
    except:
        st.title("GeoAPPS Hub")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    st.markdown("## Navigation")
    app_mode = st.selectbox(
        "Select Module",
        ["Home", "AVAzMOD", "GeoStressMOD", "PasseyTOCMOD", "RockPhysics AVO & Fluid Substitution", "WedgeMOD", "Machine Learning"],
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
# MAIN CONTENT AREA
# ===========================================
if app_mode == "Home":
    st.title("Welcome to GeoAPPS Hub developed By PhD. J.A. Guerrero Castro")
    st.markdown("---")
    
    # Developer photo section
    col_dev1, col_dev2, col_dev3 = st.columns([1,2,1])
    with col_dev2:
        try:
            developer_img = Image.open("LogoFull.png")  # Make sure this image exists in your directory
            st.image(developer_img, width=600, caption="PhD. J.A. Guerrero Castro", use_container_width=False, output_format="auto", clamp=False, channels="RGB")
        except:
            st.warning("Developer photo not found. Please add 'developer_photo.jpg' to your directory.")
    
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
            st.image(hero_img, use_container_width=True)
        except:
            st.image("https://via.placeholder.com/500x300?text=GeoAPP+Hero+Image", 
                    use_container_width=True)
    
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
            "name": "RockPhysics AVO & Fluid Substitution",
            "icon": "🪨",
            "description": "This app performs rock physics modeling and AVO analysis for brine, oil, and gas scenarios",
            "color": "#f39c12"
        },
        {
            "name": "WedgeMOD",
            "icon": "🌊",
            "description": "Advanced Seismic Wedge Modeling tool",
            "color": "#9b59b6"
        },
        {
            "name": "Machine Learning",
            "icon": "🤖",
            "description": "Advanced ML algorithms for geoscience applications Sonic Log Prediction",
            "color": "#1abc9c"
        }
    ]
    
    # Display modules in 2 rows of 3 columns
    cols = st.columns(3)
    for idx, module in enumerate(modules[:3]):
        with cols[idx]:
            st.markdown(f"""
            <div class="module-card" style="border-top: 4px solid {module['color']}">
                <h3>{module['icon']} {module['name']}</h3>
                <p>{module['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, module in enumerate(modules[3:]):
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
        "RockPhysics AVO & Fluid Substitution": ("RPTAVOmod", "main"),
        "WedgeMOD": ("WedgeModV1_st", "main"),
        "Machine Learning": ("MLalgorithms", "main")
    }
    
    if app_mode in module_mapping:
        module_info = module_mapping[app_mode]
        with st.spinner(f"Loading {app_mode} module..."):
            if not load_module(module_info[0], module_info[1]):
                st.error(f"Failed to load {app_mode} module. Please check:")
                st.error(f"- File exists: modules/{module_info[0]}.py")
                st.error(f"- Contains function: {module_info[1]}()")
                st.error(f"- No syntax errors in the module file")
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
