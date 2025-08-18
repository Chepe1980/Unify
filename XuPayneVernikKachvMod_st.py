import numpy as np
import matplotlib.pyplot as plt
from DEM_Berryman import DEM
import Gassmann
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

def is_plottable(data):
    """Check if data is plottable (not empty and contains finite values)"""
    if isinstance(data, (np.ndarray, list)):
        return len(data) > 0 and np.all(np.isfinite(data))
    return data is not None and np.isfinite(data)

def main():
    st.title("Carbonate Rock Physics Analysis with Well Log Data")
    st.markdown("""
    This app performs carbonate rock physics analysis using:
    - Xu-Payne model
    - Vernik-Kachanov model
    - Mori-Tanaka model
    - Any combination for comparison
    
    Upload your well log data and select the model(s) to use.
    """)


# Add this right after your imports
def show_guide():
    with st.expander("📘 USER GUIDE & THEORY MANUAL (Click to Expand)"):
        st.markdown("""
        ### **Rock Physics Analysis App Guide**
        
        [Rock Physics Analysis App Guide
1. Introduction

This app combines three established rock physics models (Xu-Payne, Vernik-Kachanov, and Mori-Tanaka) to analyze carbonate reservoirs using well log data. The tool helps visualize relationships between porosity, elastic properties (Vp, Vs), and fluid saturation.
2. Theoretical Background

A. Xu-Payne Model

    Designed specifically for carbonates

    Uses Differential Effective Medium (DEM) theory

    Incorporates three pore types:

        Cracks (low aspect ratio ~0.02)

        Reference pores (aspect ratio ~0.11)

        Stiff pores (aspect ratio ~0.8)

B. Vernik-Kachanov Model

    Focuses on crack density parameter

    Uses effective medium theory with critical porosity concept

    Accounts for pore shape effects on elastic moduli

C. Mori-Tanaka Model

    Estimates effective elastic properties of composites

    Handles multiple inclusion types simultaneously

    Particularly effective for heterogeneous media

3. Step-by-Step Usage Guide

1. Data Preparation

    Prepare a CSV/Excel file with these columns (names can vary):

        Depth (m/ft)

        Porosity (v/v, 0-1 scale)

        Vp (m/s) - Compressional wave velocity

        Vs (m/s) - Shear wave velocity (optional for 3D)

        Sw (v/v, 0-1 scale) - Water saturation

2. Upload & Configuration

    Upload your file using the "Upload CSV" button

    Map your columns to the required parameters in the sidebar

    The app will auto-detect common column names (e.g., "PHIE" for porosity)

3. Model Selection
Option	Description
Xu-Payne	Best for carbonate pore system analysis
Vernik-Kachanov	Ideal for fractured reservoirs
Mori-Tanaka	Good for mixed lithologies
All	Compare all models

4. Parameter Adjustment
Key parameters to customize (sidebar):

    Common: Max porosity, fluid properties

    Xu-Payne: Aspect ratios, mineral moduli

    Vernik-Kachanov: Crack density range

    Mori-Tanaka: Inclusion type and aspect ratio

5. Visualization Tools

A. 2D Plots

    Vp vs Depth: Depth trend analysis

    Vp vs Porosity: Template comparison

    Interactive Features:

        Hover for data values

        Lasso selection to highlight zones

        Color coding by water saturation

B. 3D Plot (When "All" models selected)

    Axes: Porosity (X), Vp (Y), Vs (Z)

    Shows theoretical bounds from all models

    Rotate/zoom with mouse

C. Static Comparison Plot

    Overlays all selected models

    Useful for presentations/reports

4. Interpretation Guidelines

Quality Control

    Check that well data points fall within theoretical bounds

    Points outside bounds may indicate:

        Data quality issues

        Need for parameter adjustment

        Special lithologies not accounted for

Model Selection Tips

    Use Xu-Payne for:

        Carbonates with complex pore systems

        Diagenetic studies

    Use Vernik-Kachanov for:

        Fractured reservoirs

        Stress sensitivity analysis

    Use Mori-Tanaka for:

        Mixed lithology formations

        Sand-shale sequences

5. Troubleshooting
Issue	Solution
"Error processing CSV"	Check column headers contain no special characters
Missing 3D plot	Verify Vs column is loaded and mapped
Models not appearing	Check all required parameters are set
Points outside bounds	Adjust aspect ratios/crack density
6. References

    Xu & Payne (2009) "Modeling elastic properties in carbonate rocks"

    Vernik & Kachanov (2010) "Hydraulic fracturing in shale gas reservoirs"

    Mori & Tanaka (1973) "Average stress in matrix and inclusion"]
        
        """, unsafe_allow_html=True)

# Call it at the start of your main()
def main():
    show_guide()
    # Rest of your existing code...









    
    # Model selection
    model_choice = st.sidebar.radio(
        "Select Rock Physics Model(s)",
        ["Xu-Payne", "Vernik-Kachanov", "Mori-Tanaka", "All"],
        index=0
    )

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if not uploaded_file:
        st.warning("Please upload a CSV file to proceed")
        return

    # Process CSV file with robust column selection
    try:
        df = pd.read_csv(uploaded_file)
        available_columns = df.columns.tolist()
        
        # Add index column first
        df['index'] = range(len(df))
        
        # Column selection with safe index handling
        st.sidebar.header("Data Column Selection")
        
        # Helper function for safe selection
        def safe_selectbox(label, options, priority_keys=None, default_idx=0):
            if not options:
                return None
            if priority_keys:
                priority_options = [col for col in options 
                                  if any(key.lower() in col.lower() for key in priority_keys)]
                if priority_options:
                    return st.sidebar.selectbox(label, priority_options)
            return st.sidebar.selectbox(label, options, index=min(default_idx, len(options)-1))
        
        # Select columns with intelligent defaults
        depth_col = safe_selectbox("Depth Column", available_columns, ['depth', 'dept', 'md'], 0)
        por_col = safe_selectbox("Porosity Column", available_columns, ['phi', 'por', 'phie'], 1)
        vp_col = safe_selectbox("Vp Column (m/s)", available_columns, ['vp', 'pwave', 'dtp'], 2)
        vs_col = safe_selectbox("Vs Column (m/s)", available_columns, ['vs', 'swave', 'dts'], 3)
        
        # Water saturation from remaining columns
        remaining_cols = [col for col in available_columns 
                         if col not in [depth_col, por_col, vp_col, vs_col]]
        sw_col = safe_selectbox("Water Saturation Column", remaining_cols, ['sw', 'satur'], 0)
        
        # Get data with unit conversion
        depth = df[depth_col].values
        phie = df[por_col].values
        vp = df[vp_col].values * 0.001  # convert m/s to km/s
        vs = df[vs_col].values * 0.001 if vs_col in df.columns else None
        sw = df[sw_col].values
        
        # Store Vs availability and validate data
        st.session_state.has_vs = vs is not None
        if st.session_state.has_vs:
            st.success(f"Vs data loaded! Range: {np.nanmin(vs):.2f} to {np.nanmax(vs):.2f} km/s")
        else:
            st.warning("No Vs data found - 3D plot will be limited")
        
        # Show sample data
        st.subheader("Sample Data Preview")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return

    # Common parameters
    st.sidebar.header("Common Parameters")
    phimax = st.sidebar.number_input("Maximum Porosity", value=0.4)
    Kf = st.sidebar.number_input("Fluid Bulk Modulus (GPa)", value=2.24)
    rhof = st.sidebar.number_input("Fluid Density (g/cm3)", value=0.94)

    # Xu-Payne specific parameters
    if model_choice in ["Xu-Payne", "All"]:
        st.sidebar.subheader("Xu-Payne Parameters")
        cols = st.sidebar.columns(2)
        with cols[0]:
            Km_lim = st.number_input("Limestone K (GPa)", value=77.0)
            Gm_lim = st.number_input("Limestone G (GPa)", value=32.0)
        with cols[1]:
            rhom_lim = st.number_input("Limestone ρ (g/cm3)", value=2.71)
            Km_dol = st.number_input("Dolomite K (GPa)", value=69.0)
            Gm_dol = st.number_input("Dolomite G (GPa)", value=52.0)
            rhom_dol = st.number_input("Dolomite ρ (g/cm3)", value=2.88)
        
        alpha_ref = st.sidebar.number_input("Reference Aspect Ratio", value=0.11)
        alpha_crack = st.sidebar.number_input("Crack Aspect Ratio", value=0.02)
        alpha_stiff = st.sidebar.number_input("Stiff Aspect Ratio", value=0.8)

        # Generate Xu-Payne curves
        def generate_xu_payne_curves():
            crackandref_alphas = []
            crackandref_volumes = []
            stiffandref_alphas = []
            stiffandref_volumes = []
            fractions = [0.2, 0.4, 0.6, 0.8]
            
            for fraction in fractions:
                crackandref_alphas.append([alpha_crack, alpha_ref])
                crackandref_volumes.append([fraction * phimax, (1.0 - fraction) * phimax])
                stiffandref_alphas.append([alpha_stiff, alpha_ref])
                stiffandref_volumes.append([fraction * phimax, (1.0 - fraction) * phimax])

            alphas = [[alpha_crack]] + crackandref_alphas + [[alpha_ref]] + stiffandref_alphas + [[alpha_stiff]]
            volumes = [[phimax]] + crackandref_volumes + [[phimax]] + stiffandref_volumes + [[phimax]]
            alphas += [[alpha_ref]]
            volumes += [[phimax]]

            Kms = len(alphas) * [Km_lim]
            Gms = len(alphas) * [Gm_lim]
            Kms[-1] = Km_dol
            Gms[-1] = Gm_dol

            xu_payne_curves = []

            for inclusion_alphas, inclusion_volumes, Km, Gm in zip(alphas, volumes, Kms, Gms):
                ni = len(inclusion_alphas)
                Kis = np.zeros(ni, dtype=float)
                Gis = np.zeros(ni, dtype=float)

                K, G, phi = DEM(Km, Gm, Kis, Gis, np.array(inclusion_alphas), np.array(inclusion_volumes))
                rho = (1.0 - phi) * rhom_lim + phi * rhof
                Ks = Gassmann.Ks(K, Km, Kf, phi)
                Vp = np.sqrt((Ks + 4.0 * G / 3.0) / rho)
                Vs = np.sqrt(G / rho)
                xu_payne_curves.append((phi, Vp, Vs))

            return xu_payne_curves

        xu_payne_curves = generate_xu_payne_curves()

    # Vernik-Kachanov specific parameters
    if model_choice in ["Vernik-Kachanov", "All"]:
        st.sidebar.subheader("Vernik-Kachanov Parameters")
        cols = st.sidebar.columns(2)
        with cols[0]:
            K0 = st.number_input("Matrix K (GPa)", value=75.0)
            G0 = st.number_input("Matrix G (GPa)", value=45.0)
        with cols[1]:
            rho0 = st.number_input("Matrix ρ (g/cm3)", value=2.71)
            crit_por = st.number_input("Critical Porosity", value=0.5)
        
        crack_density = st.sidebar.slider("Crack Density Range", 0.0, 1.0, (0.0, 0.8))

        # Generate Vernik-Kachanov curves
        def generate_vernik_kachanov_curves():
            try:
                porosities = np.linspace(0.01, phimax, 20)
                curves = []
                
                for cd in np.linspace(crack_density[0], crack_density[1], 5):
                    vps = []
                    vss = []
                    for phi in porosities:
                        K_eff = K0 * (1 - phi/crit_por) / (1 + 3*K0*cd/(4*G0*(1 - phi/crit_por)))
                        G_eff = G0 * (1 - phi/crit_por) / (1 + (G0*cd + 3*K0*cd/4)/(G0*(1 - phi/crit_por)))
                        
                        Ks = Gassmann.Ks(K_eff, K0, Kf, phi)
                        rho = (1 - phi) * rho0 + phi * rhof
                        Vp = np.sqrt((Ks + 4.0 * G_eff / 3.0) / rho)
                        Vs = np.sqrt(G_eff / rho)
                        vps.append(Vp)
                        vss.append(Vs)
                    
                    curves.append((porosities, np.array(vps), np.array(vss), cd))
                return curves
            except Exception as e:
                st.error(f"Error generating Vernik-Kachanov curves: {str(e)}")
                return []

        vk_curves = generate_vernik_kachanov_curves()

    # Mori-Tanaka specific parameters
    if model_choice in ["Mori-Tanaka", "All"]:
        st.sidebar.subheader("Mori-Tanaka Parameters")
        cols = st.sidebar.columns(2)
        with cols[0]:
            Km_mt = st.number_input("Matrix K (GPa) [MT]", value=75.0)
            Gm_mt = st.number_input("Matrix G (GPa) [MT]", value=45.0)
        with cols[1]:
            rhom_mt = st.number_input("Matrix ρ (g/cm3) [MT]", value=2.71)
        
        inclusion_type = st.sidebar.selectbox("Inclusion Type", ["Spheres", "Oblate Spheroids", "Needles"])
        aspect_ratio = st.sidebar.slider("Aspect Ratio", 0.01, 1.0, 0.1) if inclusion_type != "Spheres" else 1.0

        # Generate Mori-Tanaka curves
        def generate_mori_tanaka_curves():
            try:
                porosities = np.linspace(0.01, phimax, 20)
                vps = []
                vss = []
                
                for phi in porosities:
                    # Simplified Mori-Tanaka implementation
                    if inclusion_type == "Spheres":
                        # Eshelby's tensor for spheres
                        K_eff = Km_mt * (1 - phi) / (1 + phi * 3*Km_mt/(3*Km_mt + 4*Gm_mt))
                        G_eff = Gm_mt * (1 - phi) / (1 + phi * (6*Km_mt + 12*Gm_mt)/(5*(3*Km_mt + 4*Gm_mt)))
                    else:
                        # Simplified approach for non-spherical inclusions
                        alpha = aspect_ratio
                        f = phi
                        K_eff = Km_mt * (1 - f) / (1 + f * 3*Km_mt/(3*Km_mt + 4*Gm_mt*alpha))
                        G_eff = Gm_mt * (1 - f) / (1 + f * (6*Km_mt + 12*Gm_mt)/(5*(3*Km_mt + 4*Gm_mt*alpha)))
                    
                    Ks = Gassmann.Ks(K_eff, Km_mt, Kf, phi)
                    rho = (1 - phi) * rhom_mt + phi * rhof
                    Vp = np.sqrt((Ks + 4.0 * G_eff / 3.0) / rho)
                    Vs = np.sqrt(G_eff / rho)
                    vps.append(Vp)
                    vss.append(Vs)
                
                return [(porosities, np.array(vps), np.array(vss), inclusion_type + f" AR={aspect_ratio}")]
            except Exception as e:
                st.error(f"Error generating Mori-Tanaka curves: {str(e)}")
                return []

        mt_curves = generate_mori_tanaka_curves()

    # Create interactive 2D plots
    def create_2d_plots():
        try:
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=("Vp vs Depth", "Vp vs Porosity"),
                              horizontal_spacing=0.1)

            # Add well data to both subplots
            fig.add_trace(
                go.Scatter(
                    x=vp,
                    y=depth,
                    mode='markers',
                    marker=dict(color=sw, colorscale='Viridis', showscale=True,
                               cmin=0, cmax=1, colorbar=dict(title='Sw', x=0.45)),
                    customdata=np.stack((phie, sw, df['index']), axis=-1),
                    hovertemplate="<b>Depth</b>: %{y:.2f}<br>Vp: %{x:.2f} km/s<br>"
                                 "Porosity: %{customdata[0]:.2f}<br>"
                                 "Sw: %{customdata[1]:.2f}<extra></extra>",
                    name='Well Data',
                    selected=dict(marker=dict(color='red', size=10)),
                    unselected=dict(marker=dict(opacity=0.3))
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=phie,
                    y=vp,
                    mode='markers',
                    marker=dict(color=sw, colorscale='Viridis', showscale=False,
                               cmin=0, cmax=1),
                    customdata=np.stack((depth, sw, df['index']), axis=-1),
                    hovertemplate="Porosity: %{x:.2f}<br>Vp: %{y:.2f} km/s<br>"
                                 "Depth: %{customdata[0]:.2f}<br>"
                                 "Sw: %{customdata[1]:.2f}<extra></extra>",
                    name='Well Data',
                    selected=dict(marker=dict(color='red', size=10)),
                    unselected=dict(marker=dict(opacity=0.3))
                ),
                row=1, col=2
            )

            # Add model curves
            if model_choice in ["Xu-Payne", "All"] and xu_payne_curves:
                styles = ['solid', 'dash', 'dot', 'dashdot']
                colors = ['magenta', 'blue', 'red', 'green']
                labels = ['Crack', 'Reference', 'Stiff', 'Dolomite']
                
                for (phi, Vp, _), style, color, label in zip(xu_payne_curves[::3], styles, colors, labels):
                    if is_plottable(phi) and is_plottable(Vp):
                        fig.add_trace(
                            go.Scatter(
                                x=phi,
                                y=Vp,
                                mode='lines',
                                line=dict(color=color, dash=style),
                                name=f'XP: {label}',
                                showlegend=True
                            ),
                            row=1, col=2
                        )

            if model_choice in ["Vernik-Kachanov", "All"] and vk_curves:
                for phi, vp_curve, _, cd in vk_curves:
                    if is_plottable(phi) and is_plottable(vp_curve):
                        fig.add_trace(
                            go.Scatter(
                                x=phi,
                                y=vp_curve,
                                mode='lines',
                                line=dict(color='purple', width=1 + 2*cd),
                                name=f'VK: CD={cd:.1f}',
                                showlegend=True
                            ),
                            row=1, col=2
                        )

            if model_choice in ["Mori-Tanaka", "All"] and mt_curves:
                for phi, vp_curve, _, label in mt_curves:
                    if is_plottable(phi) and is_plottable(vp_curve):
                        fig.add_trace(
                            go.Scatter(
                                x=phi,
                                y=vp_curve,
                                mode='lines',
                                line=dict(color='orange', width=2),
                                name=f'MT: {label}',
                                showlegend=True
                            ),
                            row=1, col=2
                        )

            fig.update_layout(
                height=600,
                width=1200,
                title_text="Carbonate Rock Physics Analysis",
                hovermode='closest',
                dragmode='lasso',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig.update_yaxes(title_text="Depth", autorange="reversed", row=1, col=1)
            fig.update_xaxes(title_text="Vp (km/s)", row=1, col=1)
            fig.update_xaxes(title_text="Porosity", range=[0, phimax], row=1, col=2)
            fig.update_yaxes(title_text="Vp (km/s)", range=[2.0, 7.0], row=1, col=2)

            return fig
        except Exception as e:
            st.error(f"Error creating 2D plots: {str(e)}")
            return None

    # Create 3D rock physics template plot
    def create_3d_template_plot():
        try:
            if not (model_choice == "All" and xu_payne_curves and vk_curves and mt_curves):
                st.error("Need all models selected and curves generated")
                return None
                
            if not st.session_state.get('has_vs', False):
                st.error("Vs data not available")
                return None

            fig = go.Figure()

            # Add well data
            fig.add_trace(
                go.Scatter3d(
                    x=phie,
                    y=vp,
                    z=vs,
                    mode='markers',
                    marker=dict(
                        color=sw,
                        colorscale='Viridis',
                        size=4,
                        opacity=0.8,
                        colorbar=dict(title='Sw')
                    ),
                    name='Well Data'
                )
            )

            # Add Xu-Payne templates
            styles = ['solid', 'dash', 'dot', 'dashdot']
            colors = ['magenta', 'blue', 'red', 'green']
            labels = ['Crack', 'Reference', 'Stiff', 'Dolomite']
            
            for (phi, Vp, Vs), style, color, label in zip(xu_payne_curves[::3], styles, colors, labels):
                if is_plottable(phi) and is_plottable(Vp) and is_plottable(Vs):
                    fig.add_trace(
                        go.Scatter3d(
                            x=phi,
                            y=Vp,
                            z=Vs,
                            mode='lines',
                            line=dict(color=color, width=4, dash=style),
                            name=f'XP: {label}'
                        )
                    )

            # Add Vernik-Kachanov templates
            for phi, vp_curve, vs_curve, cd in vk_curves:
                if is_plottable(phi) and is_plottable(vp_curve) and is_plottable(vs_curve):
                    fig.add_trace(
                        go.Scatter3d(
                            x=phi,
                            y=vp_curve,
                            z=vs_curve,
                            mode='lines',
                            line=dict(color='purple', width=2 + 2*cd),
                            name=f'VK: CD={cd:.1f}'
                        )
                    )

            # Add Mori-Tanaka templates
            for phi, vp_curve, vs_curve, label in mt_curves:
                if is_plottable(phi) and is_plottable(vp_curve) and is_plottable(vs_curve):
                    fig.add_trace(
                        go.Scatter3d(
                            x=phi,
                            y=vp_curve,
                            z=vs_curve,
                            mode='lines',
                            line=dict(color='orange', width=4),
                            name=f'MT: {label}'
                        )
                    )

            fig.update_layout(
                title='3D Rock Physics Template Space',
                scene=dict(
                    xaxis_title='Porosity',
                    yaxis_title='Vp (km/s)',
                    zaxis_title='Vs (km/s)',
                    xaxis=dict(range=[0, phimax]),
                    yaxis=dict(range=[2.0, 7.0]),
                    zaxis=dict(range=[1.0, 4.0])
                ),
                width=1000,
                height=800,
                margin=dict(r=20, b=10, l=10, t=40)
            )
            return fig
            
        except Exception as e:
            st.error(f"3D plot error: {str(e)}")
            return None

    # Create and display 2D plots
    st.subheader("2D Analysis")
    fig_2d = create_2d_plots()
    if fig_2d:
        st.plotly_chart(fig_2d, use_container_width=True)
    else:
        st.warning("Could not generate 2D plots")

    # Create and display 3D plot when all models are selected
    if model_choice == "All":
        st.subheader("3D Rock Physics Template Space")
        if st.session_state.get('has_vs', False):
            fig_3d = create_3d_template_plot()
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.error("Failed to create 3D plot. Check console for errors.")
        else:
            st.warning("3D plot requires Vs data. Please select a valid shear wave velocity column.")

    # Create static comparison
    if model_choice == "All" and xu_payne_curves and vk_curves and mt_curves:
        try:
            st.subheader("Template Comparison")
            plt.figure(figsize=(10,6))
            
            # Plot data
            plt.scatter(phie, vp, c=sw, s=5, cmap='viridis', alpha=0.6, vmin=0, vmax=1)
            
            # Plot Xu-Payne curves
            styles = [':', '-', '--', '-']
            colors = ['magenta', 'blue', 'red', 'green']
            for (phi, Vp, _), style, color in zip(xu_payne_curves[::3], styles, colors):
                if is_plottable(phi) and is_plottable(Vp):
                    plt.plot(phi, Vp, style, color=color, linewidth=2, label=f'XP: {color}')
            
            # Plot Vernik-Kachanov curves
            for phi, vp_curve, _, cd in vk_curves:
                if is_plottable(phi) and is_plottable(vp_curve):
                    plt.plot(phi, vp_curve, '--', color='purple', alpha=0.5, linewidth=1+cd*2,
                            label=f'VK: CD={cd:.1f}' if cd == vk_curves[0][3] else "")
            
            # Plot Mori-Tanaka curves
            for phi, vp_curve, _, label in mt_curves:
                if is_plottable(phi) and is_plottable(vp_curve):
                    plt.plot(phi, vp_curve, '-', color='orange', linewidth=2, label=f'MT: {label}')
            
            plt.colorbar(label='Sw')
            plt.xlabel("Porosity")
            plt.ylabel("Vp (km/s)")
            plt.ylim(2.0, 7.0)
            plt.xlim(0.0, phimax)
            plt.grid()
            plt.title("Rock Physics Templates Comparison")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error creating static plot: {str(e)}")

if __name__ == '__main__':
    main()
