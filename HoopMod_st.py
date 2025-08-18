import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from io import StringIO






# Available colormaps
colormaps = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 
             'Cool', 'RdYlBu', 'Rainbow', 'Jet']

def kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, theta_deg, azimuth_deg, dip_deg, deviation_deg, r=1.0):
    theta = np.radians(theta_deg)
    azimuth = np.radians(azimuth_deg)
    dip = np.radians(dip_deg)
    deviation = np.radians(deviation_deg)
    
    # Stress transformation (3D rotation)
    R = np.array([
        [np.cos(deviation)*np.cos(azimuth), -np.sin(azimuth), np.sin(deviation)*np.cos(azimuth)],
        [np.cos(deviation)*np.sin(azimuth), np.cos(azimuth), np.sin(deviation)*np.sin(azimuth)],
        [-np.sin(deviation), 0, np.cos(deviation)]
    ])
    
    sigma_global = np.diag([Shmax, Shmin, Sv])
    sigma_local = R.T @ sigma_global @ R
    
    # Kirsch equations
    a = 1.0  # Borehole radius
    radial = (sigma_local[0,0] + sigma_local[1,1])/2 * (1 - a**2/r**2) + \
             (sigma_local[0,0] - sigma_local[1,1])/2 * (1 - 4*a**2/r**2 + 3*a**4/r**4) * np.cos(2*theta) + \
             sigma_local[0,1] * (1 - 4*a**2/r**2 + 3*a**4/r**4) * np.sin(2*theta) + \
             PP * a**2/r**2 - wellbore_pressure * a**2/r**2
    
    hoop = (sigma_local[0,0] + sigma_local[1,1])/2 * (1 + a**2/r**2) - \
           (sigma_local[0,0] - sigma_local[1,1])/2 * (1 + 3*a**4/r**4) * np.cos(2*theta) - \
           sigma_local[0,1] * (1 + 3*a**4/r**4) * np.sin(2*theta) - \
           PP * a**2/r**2 + wellbore_pressure * a**2/r**2
    
    shear = -(sigma_local[0,0] - sigma_local[1,1])/2 * (1 + 2*a**2/r**2 - 3*a**4/r**4) * np.sin(2*theta) + \
            sigma_local[0,1] * (1 + 2*a**2/r**2 - 3*a**4/r**4) * np.cos(2*theta)
    
    return radial, hoop, shear

def plot_well_logs(well_log, selected_depth):
    """Plot well logs with highlighted selected depth using Plotly"""
    fig = make_subplots(rows=1, cols=4, subplot_titles=(
        'Sv (psi)', 'Shmin(psi)',
        'Shmax (psi)', 'Pp (psi)'
    ))
    
    # Find closest depth index
    idx = (well_log['Depth'] - selected_depth).abs().idxmin()
    selected_values = well_log.iloc[idx]
    
    # Plot each log
    logs = ['Sv', 'Shmin', 'Shmax', 'PP']
    titles = ['Sv (psi)', 'Shmin (psi)', 'Shmax (psi)', 'PP (psi)']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (log, title, color) in enumerate(zip(logs, titles, colors), 1):
        fig.add_trace(
            go.Scatter(
                x=well_log[log],
                y=well_log['Depth'],
                mode='lines',
                line=dict(color=color, width=2),
                name=title,
                hovertemplate="Depth: %{y:.1f}m<br>Value: %{x:.1f}psi<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Add selected depth marker
        fig.add_trace(
            go.Scatter(
                x=[selected_values[log]],
                y=[selected_depth],
                mode='markers',
                marker=dict(color='red', size=10),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=i
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update y-axes
    for i in range(1, 5):
        fig.update_yaxes(title_text="Depth (m)", row=1, col=i, autorange="reversed")
    
    return fig

def update_plots(Sv, Shmin, Shmax, PP, wellbore_pressure, azimuth, dip, deviation, selected_cmap):
    # Calculate stresses
    theta_vals = np.linspace(0, 360, 360)
    radial, hoop, shear = kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                                       theta_vals, azimuth, dip, deviation, r=1.0)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "polar"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "scene"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "Stress Components (Polar View)",
            "Hoop Stress vs Angle",
            "All Stress Components",
            "2D Hoop Stress Distribution",
            "3D Hoop Stress Distribution",
            "Stress Magnitude"
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    # 1. Polar Plot
    fig.add_trace(
        go.Scatterpolar(
            r=hoop,
            theta=theta_vals,
            mode='lines',
            name='Hoop Stress',
            line=dict(color='red', width=2),
            hovertemplate="Angle: %{theta:.1f}°<br>Hoop Stress: %{r:.1f}psi<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatterpolar(
            r=radial,
            theta=theta_vals,
            mode='lines',
            name='Radial Stress',
            line=dict(color='blue', width=2),
            hovertemplate="Angle: %{theta:.1f}°<br>Radial Stress: %{r:.1f}psi<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatterpolar(
            r=shear,
            theta=theta_vals,
            mode='lines',
            name='Shear Stress',
            line=dict(color='green', width=2),
            hovertemplate="Angle: %{theta:.1f}°<br>Shear Stress: %{r:.1f}psi<extra></extra>"
        ),
        row=1, col=1
    )
    fig.update_polars(
        angularaxis_direction='clockwise',
        angularaxis_rotation=90
    )
    
    # 2. Hoop Stress (Cartesian)
    fig.add_trace(
        go.Scatter(
            x=theta_vals,
            y=hoop,
            mode='lines',
            line=dict(color='red', width=2),
            hovertemplate="Angle: %{x:.1f}°<br>Hoop Stress: %{y:.1f}psi<extra></extra>",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. All Stress Components
    fig.add_trace(
        go.Scatter(
            x=theta_vals,
            y=hoop,
            mode='lines',
            name='Hoop Stress',
            line=dict(color='red', width=2),
            hovertemplate="Angle: %{x:.1f}°<br>Hoop Stress: %{y:.1f}psi<extra></extra>"
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(
            x=theta_vals,
            y=radial,
            mode='lines',
            name='Radial Stress',
            line=dict(color='blue', width=2),
            hovertemplate="Angle: %{x:.1f}°<br>Radial Stress: %{y:.1f}psi<extra></extra>"
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(
            x=theta_vals,
            y=shear,
            mode='lines',
            name='Shear Stress',
            line=dict(color='green', width=2),
            hovertemplate="Angle: %{x:.1f}°<br>Shear Stress: %{y:.1f}psi<extra></extra>"
        ),
        row=1, col=3
    )
    
    # 4. 2D Hoop Stress Distribution
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    _, Hoop_2d, _ = kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                                  np.degrees(Theta), azimuth, dip, deviation, R)
    Hoop_2d[R < 1.0] = np.nan
    
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=Hoop_2d,
            colorscale=selected_cmap,
            showscale=True,
            colorbar=dict(title='Hoop Stress (psi)'),
            hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Stress: %{z:.1f}psi<extra></extra>",
            opacity=0.9  # Slightly reduce opacity to make arrow more visible
        ),
        row=2, col=1
    )
    
    # Add borehole outline
    theta_circle = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta_circle),
            y=np.sin(theta_circle),
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )

    # Add arrow showing Shmin direction (fixed implementation)
    arrow_length = 2.5
    arrow_angle = np.radians(azimuth + 90)  # Shmin is perpendicular to Shmax
    arrow_x = arrow_length * np.cos(arrow_angle)
    arrow_y = arrow_length * np.sin(arrow_angle)


      # Add arrow as a separate trace for better control
    fig.add_trace(
        go.Scatter(
            x=[0, arrow_x],
            y=[0, arrow_y],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(
                symbol='arrow',
                size=15,
                angleref='previous',
                color='red'
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )









  
    
    fig.add_annotation(
        x=arrow_x,
        y=arrow_y,
        ax=0,
        ay=0,
        xref="x4",
        yref="y4",
        axref="x4",
        ayref="y4",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor='white',
        row=2, col=1
    )
    
    fig.add_annotation(
        x=arrow_x * 1.2,
        y=arrow_y * 1.2,
        text="Shmax",
        showarrow=False,
        font=dict(color='white', size=14, family="Arial Black"),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='white',
        borderwidth=1,
        borderpad=4,
        row=2, col=1
    )
    




  
    








    # 5. 3D Surface Plot
    r_3d = np.linspace(1, 3, 50)
    theta_3d = np.radians(np.linspace(0, 360, 50))
    R_3d, Theta_3d = np.meshgrid(r_3d, theta_3d)
    X_3d = R_3d * np.cos(Theta_3d)
    Y_3d = R_3d * np.sin(Theta_3d)
    _, Hoop_3d, _ = kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                                  np.degrees(Theta_3d), azimuth, dip, deviation, R_3d)
    
    fig.add_trace(
        go.Surface(
            x=X_3d,
            y=Y_3d,
            z=Hoop_3d,
            colorscale=selected_cmap,
            showscale=True,
            colorbar=dict(title='Hoop Stress (MPa)'),
            hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Stress: %{z:.1f}psi<extra></extra>"
        ),
        row=2, col=2
    )
    
    # 6. Stress Magnitude
    stress_magnitude = np.sqrt(hoop**2 + radial**2 + shear**2)
    fig.add_trace(
        go.Scatter(
            x=theta_vals,
            y=stress_magnitude,
            mode='lines',
            name='Stress Magnitude',
            line=dict(color='black', width=2),
            hovertemplate="Angle: %{x:.1f}°<br>Magnitude: %{y:.1f}psi<extra></extra>"
        ),
        row=2, col=3
    )
    fig.add_trace(
        go.Scatter(
            x=theta_vals,
            y=hoop,
            mode='lines',
            name='Hoop Stress',
            line=dict(color='red', width=1),
            opacity=0.3,
            hovertemplate="Angle: %{x:.1f}°<br>Hoop Stress: %{y:.1f}psi<extra></extra>",
            showlegend=False
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Borehole Stress Analysis",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Angle (degrees)", row=1, col=2)
    fig.update_yaxes(title_text="Hoop Stress (psi)", row=1, col=2)
    fig.update_xaxes(title_text="Angle (degrees)", row=1, col=3)
    fig.update_yaxes(title_text="Stress (psi)", row=1, col=3)
    fig.update_xaxes(title_text="X (m)", row=2, col=1)
    fig.update_yaxes(title_text="Y (m)", row=2, col=1)
    fig.update_xaxes(title_text="Angle (degrees)", row=2, col=3)
    fig.update_yaxes(title_text="Stress (psi)", row=2, col=3)
    
    # Update 3D scene
    fig.update_scenes(
        aspectmode='cube',
        row=2, col=2
    )
    
    return fig

def load_well_log(uploaded_file):
    """Load well log data from CSV or Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        required_columns = {'Depth', 'Sv', 'Shmin', 'Shmax', 'PP'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"Missing required columns: {missing}")
            return None
        return df.sort_values('Depth')
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide", page_title="Stress Wellbore Analysis")
    st.title("📊 Borehole Hoop Stress Visualization Model Tool")
    st.markdown("""
    <style>
    .stSlider>div {padding: 0.5rem 0;}
    .st-b7 {font-size: 1.2rem !important;}
    .st-cb {background-color: #f0f2f6;}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'calculate' not in st.session_state:
        st.session_state.calculate = False
    
    # File upload section
    with st.expander("📁 Upload Well Log Data", expanded=True):
        uploaded_file = st.file_uploader("Upload well log (CSV or Excel)", 
                                       type=["csv", "xlsx", "xls"],
                                       help="Required columns: Depth, Sv, Shmin, Shmax, PP")
    
    if uploaded_file is not None:
        well_log = load_well_log(uploaded_file)
        
        if well_log is not None:
            st.success(f"✅ Successfully loaded well log with {len(well_log)} data points")
            
            # Depth selection
            min_depth = float(well_log['Depth'].min())
            max_depth = float(well_log['Depth'].max())
            selected_depth = st.slider(
                'Select Depth (m)', 
                min_value=min_depth, 
                max_value=max_depth, 
                value=(min_depth + max_depth)/2,
                step=0.1,
                format="%.1f"
            )
            
            # Display well logs with highlighted depth
            st.subheader("Well Logs at Selected Depth")
            well_log_fig = plot_well_logs(well_log, selected_depth)
            st.plotly_chart(well_log_fig, use_container_width=True)
            
            # Find closest depth in log
            idx = (well_log['Depth'] - selected_depth).abs().idxmin()
            log_data = well_log.iloc[idx]
            
            st.markdown(f"**Selected Depth:** {log_data['Depth']:.2f}m")
            
            # Main parameter controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                Sv = st.number_input('Sv (psi)', value=float(log_data['Sv']), min_value=0.0, step=1.0)
            with col2:
                Shmin = st.number_input('Shmin (psi)', value=float(log_data['Shmin']), min_value=0.0, step=1.0)
            with col3:
                Shmax = st.number_input('Shmax (psi)', value=float(log_data['Shmax']), min_value=0.0, step=1.0)
            with col4:
                PP = st.number_input('Pore Pressure (psi)', value=float(log_data['PP']), min_value=0.0, step=1.0)
            
            # Wellbore parameters in sidebar
            with st.sidebar:
                st.header("⚙️ Wellbore Parameters")
                wellbore_pressure = st.slider('Wellbore Pressure (psi)', 0.0, 100.0, 10.0, 1.0)
                azimuth = st.slider('Azimuth (°)', 0, 360, 0, 1)
                dip = st.slider('Dip (°)', 0, 90, 0, 1)
                deviation = st.slider('Deviation (°)', 0, 90, 0, 1)
                selected_cmap = st.selectbox('Color Map:', colormaps, index=0)
                
                if st.button('🚀 Calculate Stresses', type="primary"):
                    st.session_state.calculate = True
                
                if st.button('💾 Export Data to CSV'):
                    theta_vals = np.linspace(0, 360, 360)
                    radial, hoop, shear = kirsch_stresses(
                        Sv, Shmin, Shmax, PP, wellbore_pressure,
                        theta_vals, azimuth, dip, deviation
                    )
                    df = pd.DataFrame({
                        'Theta (deg)': theta_vals,
                        'Hoop Stress (psi)': hoop,
                        'Radial Stress (psi)': radial,
                        'Shear Stress (psi)': shear,
                        'Stress Magnitude (psi)': np.sqrt(hoop**2 + radial**2 + shear**2)
                    })
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f'stresses_{selected_depth:.1f}m.csv',
                        mime='text/csv'
                    )

            if st.session_state.calculate:
                with st.spinner('Calculating stresses...'):
                    stress_fig = update_plots(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                                           azimuth, dip, deviation, selected_cmap)
                    st.plotly_chart(stress_fig, use_container_width=True)

if __name__ == "__main__":
    main()




