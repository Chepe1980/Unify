import streamlit as st

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Module",
    ["Home", "AVAzMOD", "GeoStressMOD","PasseyTOCMOD", 
     "Machine Learning Algorithms"])

if app_mode == "Geosciences Application Home":
    st.title("GeoAPPS Hub")
    st.write("Select Module from the sidebar to explore.")
    
elif app_mode == "Genetic Algorithm":
    # Import or include code for Algorithm 1
    from AVAzAPP import main
    main()
    
elif app_mode == "Convolutional Neural Network Algorithm":
    # Import or include code for Algorithm 2
    from GeoMechApp import main
    main()

elif app_mode == "Long Short Term memory Algorithm":
    # Import or include code for Algorithm 2
    from TocApp_st import main
    main()

elif app_mode == "Random Forest Algorithm":
    # Import or include code for Algorithm 2
    from MLalgorithms import main
    main()


