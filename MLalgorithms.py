import streamlit as st

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the Machine learninf algorithm",
    ["Home", "Genetic Algorithm", "Convolutional Neural Network Algorithm", "Long Short Term memory Algorithm", 
     "Random Forest Algorithm", "Probabilistic Neural Network Algorithm"])

if app_mode == "Home":
    st.title("Machine Learning Algorithms Hub")
    st.write("Select an algorithm from the sidebar to explore.")
    
elif app_mode == "Genetic Algorithm":
    # Import or include code for Algorithm 1
    from GenApp_st import main
    main()
    
elif app_mode == "Convolutional Neural Network Algorithm":
    # Import or include code for Algorithm 2
    from CnnApp_st2 import main
    main()

elif app_mode == "Long Short Term memory Algorithm":
    # Import or include code for Algorithm 2
    from LstmApp_st import main
    main()

elif app_mode == "Random Forest Algorithm":
    # Import or include code for Algorithm 2
    from RfaApp_st import main
    main()

    
elif app_mode == "Probabilistic Neural Network Algorithm":
    # Import or include code for Algorithm 2
    from PnnApp_st import main
    main()
    

    
