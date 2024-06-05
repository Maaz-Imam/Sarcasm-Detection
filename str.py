import streamlit as st
import pandas as pd
from prediction import *

# Title of the Streamlit app
st.title('Sarcasm Detection')

# Sidebar options for input method
st.sidebar.title('Input Method')
input_method = st.sidebar.radio('Choose an input method:', ('Upload CSV', 'Enter Text Prompt'))

if input_method == 'Upload CSV':
    # File upload for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        
        # Display the uploaded data
        st.write("Uploaded CSV:")
        st.write(data)

        # Feature selection
        selected_features = st.multiselect('Select features to use:', ['explicit', 'implicit'])

        # Placeholder for backend processing
        st.write("Processing...")
        result = mainn(data, selected_features)

        def get_row_color(row):
            if row['Sarcasm Tag'] == 'yes':
                return ['background-color: lightgreen'] * len(row)
            elif row['Sarcasm Tag'] == 'no':
                return ['background-color: lightcoral'] * len(row)
            else:
                return [''] * len(row)

        # Apply colors to the DataFrame
        styled_df = result.style.apply(get_row_color, axis=1)

        # Display the styled DataFrame
        st.write("Results:")
        st.write(styled_df)
        
elif input_method == 'Enter Text Prompt':
    # Text input for a single prompt
    text_prompt = st.text_area("Enter your text prompt here:")

    if st.button('Submit'):
        if text_prompt:
            # Feature selection
            selected_features = st.multiselect('Select features to use:', ['explicit', 'implicit'])

            # Placeholder for backend processing
            result = predict_sarcasm(text_prompt, selected_features)

            st.write("Processing...")
            
            # Display the results
            st.write("Results:")
            st.write(result)
        else:
            st.write("Please enter a text prompt.")
