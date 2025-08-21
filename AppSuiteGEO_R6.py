import streamlit as st
from PIL import Image
import importlib.util
import sys
from pathlib import Path
import os
import time
import requests
from io import BytesIO
from typing import Dict, Tuple, Optional
import urllib.parse

# ===========================================
# CONFIGURATION
# ===========================================
# GitHub repository configuration - UPDATE THIS WITH YOUR ACTUAL REPO URL
GITHUB_REPO_RAW_URL = "https://raw.githubusercontent.com/Chepe1980/Unify/main/"

# Module configuration with direct GitHub URLs
MODULE_CONFIG = {
    "AVAzMOD": {
        "file": "AVAzAPP.py",
        "function": "main",
        "description": "Azimuthal Velocity Analysis for fracture characterization and Fluid Substitution in shale rocks",
        "color": "#3498db",
        "logo": f"{GITHUB_REPO_RAW_URL}avazmod_logo.png"
    },
    "GeoStressMOD": {
        "file": "GeoMechApp.py", 
        "function": "main",
        "description": "Geomechanical stress analysis and modeling of Hoop Stress wellbore",
        "color": "#e74c3c",
        "logo": f"{GITHUB_REPO_RAW_URL}geostress_logo.png"
    },
    "PasseyTOCMOD": {
        "file": "TocApp_st.py",
        "function": "main",
        "description": "Total Organic Carbon calculation using ΔLogR method of Passey´s method",
        "color": "#2ecc71",
        "logo": f"{GITHUB_REPO_RAW_URL}passey_logo.png"
    },
    "RockPhysics AVO & Fluid Substitution": {
        "file": "RPTAVOmod.py",
        "function": "main", 
        "description": "This app performs rock physics modeling and AVO analysis for brine, oil, and gas scenarios",
        "color": "#f39c12",
        "logo": f"{GITHUB_REPO_RAW_URL}RPTlogo.png"
    },
    "WedgeMOD": {
        "file": "WedgeModV1_st.py",
        "function": "main",
        "description": "Advanced Seismic Wedge Modeling tool",
        "color": "#9b59b6",
        "logo": f"{GITHUB_REPO_RAW_URL}wedgemod_logo.png"
    },
    "Machine Learning": {
        "file": "MLalgorithms.py",
        "function": "main",
        "description": "Advanced ML algorithms for geoscience applications Sonic Log Prediction",
        "color": "#1abc9c",
        "logo": f"{GITHUB_REPO_RAW_URL}ml_logo.png"
    },
    "HoopMod": {
        "file": "HoopMod_st.py",
        "function": "main",
        "description": "Hoop Stress analysis and visualization for wellbore stability",
        "color": "#16a085",
        "logo": f"{GITHUB_REPO_RAW_URL}hoopmod_logo.png"
    },
    "ShaleRPTMod": {
        "file": "ShaleRPTMod_st.py",
        "function": "main",
        "description": "Shale Rock Physics Template analysis and modeling",
        "color": "#8e44ad",
        "logo": f"{GITHUB_REPO_RAW_URL}shalerpt_logo.png"
    },
    "CarbonateRPTMod": {
        "file": "XuPayneVernikKachvMod_st.py",
        "function": "main",
        "description": "Rock physics modeling using Xu-Payne, Vernik-Kachanov methods",
        "color": "#d35400",
        "logo": f"{GITHUB_REPO_RAW_URL}xupayne_logo.png"
    }
}

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
# UTILITY FUNCTIONS FOR GITHUB CONTENT
# ===========================================
def load_image_from_url(url: str, width: int = None) -> Image.Image:
    """Load image from URL with error handling"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if width:
            # Calculate height to maintain aspect ratio
            aspect_ratio = img.height / img.width
            height = int(width * aspect_ratio)
            img = img.resize((width, height))
        return img
    except Exception as e:
        st.warning(f"Could not load image from {url}: {e}")
        return None

def download_module_from_github(module_name: str, module_info: dict) -> bool:
    """Download module from GitHub if not available locally"""
    try:
        module_url = f"{GITHUB_REPO_RAW_URL}{module_info['file']}"
        response = requests.get(module_url)
        response.raise_for_status()
        
        # Save module locally
        os.makedirs("modules", exist_ok=True)
        with open(f"modules/{module_info['file']}", "wb") as f:
            f.write(response.content)
        
        st.success(f"Downloaded {module_name} module from GitHub")
        return True
    except Exception as e:
        st.error(f"Failed to download {module_name} module: {e}")
        return False

def create_placeholder_module(module_path: Path, module_name: str) -> None:
    """Creates a placeholder module if it doesn't exist"""
    placeholder_code = f'''# {module_name} placeholder
import streamlit as st
import pandas as pd
import numpy as np

def main():
    """Placeholder module for {module_name}"""
    st.title("{module_name.replace('_', ' ').title()} Module")
    st.image("https://via.placeholder.com/800x400?text={module_name}+Module", 
             use_container_width=True)
    st.warning("This is a placeholder module")
    st.info("The actual implementation will go here")
    
    if st.button("Sample Button"):
        st.success("Button clicked!")

    st.markdown("### Sample DataFrame")
    df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'])
    st.dataframe(df)
    
    # Add some sample visualizations
    st.markdown("### Sample Chart")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Velocity', 'Density', 'Impedance'])
    st.line_chart(chart_data)
'''
    with open(module_path, 'w') as f:
        f.write(placeholder_code)

def load_module(module_name: str, function_name: str = "main") -> bool:
    """Load module from GitHub or local file"""
    try:
        module_info = MODULE_CONFIG.get(module_name)
        if not module_info:
            st.error(f"No configuration found for module: {module_name}")
            return False
            
        module_file = module_info["file"]
        module_path = Path("modules") / module_file
        
        # Download from GitHub if not available locally
        if not module_path.exists():
            if not download_module_from_github(module_name, module_info):
                # If download fails, create a placeholder
                create_placeholder_module(module_path, module_name)
                st.info(f"Created placeholder module at {module_path}")
        
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
            transition: transform 0.2s, box-shadow 0.2s;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .module-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(52, 152, 219, 0.4);
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
        
        /* Module logo styling */
        .module-logo {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            object-fit: cover;
            margin: 0 auto 10px auto;
            display: block;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        }
        
        /* Custom footer */
        .footer {
            text-align: center;
            padding: 1rem;
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 2rem;
            border-top: 1px solid #e0e0e0;
        }
        
        /* GitHub logo styling */
        .github-logo {
            font-size: 1.5rem;
            margin-right: 0.5rem;
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
        logo_url = f"{GITHUB_REPO_RAW_URL}logoApps.png"
        logo = load_image_from_url(logo_url)
        if logo:
            st.image(logo, use_container_width=True)
        else:
            st.title("🌐 GeoAPPS Hub")
            st.markdown("**Advanced Geoscience Tools**")
    except:
        st.title("🌐 GeoAPPS Hub")
        st.markdown("**Advanced Geoscience Tools**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    st.markdown("## 📂 Navigation")
    app_mode = st.selectbox(
        "Select Module",
        ["Home"] + list(MODULE_CONFIG.keys()),
        label_visibility="collapsed"
    )
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    total_modules = len(MODULE_CONFIG)
    available_modules = sum(1 for mod in MODULE_CONFIG.values() if Path("modules").exists() and Path(f"modules/{mod['file']}").exists())
    st.markdown(f"**Total Modules:** {total_modules}")
    st.markdown(f"**Available:** {available_modules}")
    
    # Refresh button
    if st.button("🔄 Refresh Modules"):
        st.experimental_rerun()
    
    # GitHub link
    st.markdown("---")
    st.markdown(f"""
    **🔗 Repository Link**  
    [<span class='github-logo'>🐙</span>View on GitHub](https://github.com/Chepe1980/Unify)
    """, unsafe_allow_html=True)
    
    # About section
    st.markdown("---")
    st.markdown("""
    **📋 About GeoAPPS Hub**  
    Integrated geoscience tools for advanced analysis  
    
    **Version:** 1.0.0  
    **Developer:** PhD. J.A. Guerrero Castro  
    
    [📧 Contact Support](mailto:alfredoguerrero1980@gmail.com)
    """)

# ===========================================
# MAIN CONTENT AREA
# ===========================================
if app_mode == "Home":
    st.title("Welcome to GeoAPPS Hub")
    st.markdown("**Developed By PhD. J.A. Guerrero Castro**")
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚀 Integrated Geoscience Solutions
        
        Access specialized tools for advanced geoscience analysis, 
        modeling, and machine learning applications.
        
        **Features:**
        - 🎯 Advanced seismic analysis tools
        - 📈 Rock physics modeling
        - 🤖 Machine learning algorithms
        - 📊 Visualization and reporting
        - 🔄 Fluid substitution modeling
        """)
        
    with col2:
        hero_url = f"{GITHUB_REPO_RAW_URL}logoApps.png"
        hero_img = load_image_from_url(hero_url)
        if hero_img:
            st.image(hero_img, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/500x300?text=GeoAPP+Hero+Image", use_container_width=True)
    
    # Instructions section
    st.markdown("---")
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. Select a module from the sidebar navigation
    2. The app will automatically download the module from GitHub if needed
    3. Use the module's interface to perform your analysis
    4. Return to the home page to access other modules
    """)
    
    # Module showcase
    st.markdown("## 🗂️ Available Modules")
    st.markdown("---")
    
    # Display modules in a grid
    cols = st.columns(3)
    for idx, (module_name, module_info) in enumerate(MODULE_CONFIG.items()):
        with cols[idx % 3]:
            # Check if module exists
            module_exists = Path("modules").exists() and Path(f"modules/{module_info['file']}").exists()
            status_icon = "✅" if module_exists else "⏳"
            
            st.markdown(f"""
            <div class="module-card" style="border-top: 4px solid {module_info['color']};text-align:center;">
                <img src="{module_info['logo']}" class="module-logo" onerror="this.src='https://via.placeholder.com/60x60?text={module_name[0]}'">
                <h3>{module_name} {status_icon}</h3>
                <p>{module_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Module loading logic
    st.title(f"🧩 {app_mode}")
    st.markdown("---")
    
    # Show module info
    module_info = MODULE_CONFIG.get(app_mode)
    if module_info:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Description:** {module_info['description']}")
            st.markdown(f"**File:** `{module_info['file']}`")
        with col2:
            logo_img = load_image_from_url(module_info['logo'], width=80)
            if logo_img:
                st.image(logo_img)
            else:
                st.markdown(f"<div style='width:80px; height:80px; border-radius:50%; background:{module_info['color']}; display:flex; align-items:center; justify-content:center; font-size:24px; color:white;'>{app_mode[0]}</div>", unsafe_allow_html=True)
        
        # Load module with progress indicator
        with st.spinner(f"Loading {app_mode} module..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate loading
                progress_bar.progress(i + 1)
            
            if load_module(app_mode, module_info["function"]):
                st.success(f"{app_mode} loaded successfully!")
            else:
                st.error(f"Failed to load {app_mode}")
                st.info("""
                **Troubleshooting Tips:**
                1. Check your internet connection
                2. Verify the GitHub repository URL is correct
                3. Ensure the module file exists in the repository
                """)
    else:
        st.error("Module configuration error: Unknown module selected")

# ===========================================
# FOOTER
# ===========================================
st.markdown("""
<div class="footer">
    © 2025 GeoAPPS Hub | Developed by PhD J.A Guerrero Castro | v1.0.0
</div>
""", unsafe_allow_html=True)
