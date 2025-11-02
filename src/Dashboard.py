import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

# Backend endpoint
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# Make sure you have wine_model.pkl file in FastAPI_Labs/src/model folder
FASTAPI_WINE_MODEL_LOCATION = Path(__file__).resolve().parents[1] / 'model' / 'wine_model.pkl'

# Streamlit logger
LOGGER = get_logger(__name__)

def run():
    # Set the main dashboard page browser tab title and icon
    st.set_page_config(
        page_title="Wine Classification Demo",
        page_icon="🍷",
    )

    # Build the sidebar first
    with st.sidebar:
        # Check backend status
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            if backend_request.status_code == 200:
                st.success("Backend online ✅")
            else:
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            LOGGER.error("Backend offline 😱")
            st.error("Backend offline 😱")

        st.info("Upload your Wine dataset test JSON file")
        
        # File uploader for JSON test input
        test_input_file = st.file_uploader('Upload test prediction file', type=['json'])

        if test_input_file:
            st.write('Preview file')
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)
            st.session_state["IS_JSON_FILE_AVAILABLE"] = True
        else:
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False

        # Predict button
        predict_button = st.button('Predict')

    # Dashboard body
    st.write("# Wine Classification")

    if predict_button:
        if "IS_JSON_FILE_AVAILABLE" in st.session_state and st.session_state["IS_JSON_FILE_AVAILABLE"]:
            if FASTAPI_WINE_MODEL_LOCATION.is_file():
                client_input = test_input_data
                try:
                    result_container = st.empty()
                    with st.spinner('Predicting...'):
                        predict_wine_response = requests.post(
                            f'{FASTAPI_BACKEND_ENDPOINT}/predict', json=client_input
                        )

                    if predict_wine_response.status_code == 200:
                        wine_content = json.loads(predict_wine_response.content)
                        start_sentence = "The wine class predicted is: "
                        
                        if wine_content["response"] == 0:
                            result_container.success(f"{start_sentence} Class 0 (Cultivar 0)")
                        elif wine_content["response"] == 1:
                            result_container.success(f"{start_sentence} Class 1 (Cultivar 1)")
                        elif wine_content["response"] == 2:
                            result_container.success(f"{start_sentence} Class 2 (Cultivar 2)")
                        else:
                            result_container.error("Unknown class returned")
                            LOGGER.error("Unexpected prediction output")
                    else:
                        st.write("Status code:", predict_wine_response.status_code)
                        st.write("Response body:", predict_wine_response.text)
                        st.toast(f':red[Status from server: {predict_wine_response.status_code}. Refresh page and check backend status]', icon="🔴")
                except Exception as e:
                    st.toast(':red[Problem with backend. Refresh page and check backend status]', icon="🔴")
                    LOGGER.error(e)
            else:
                LOGGER.warning('wine_model.pkl not found in FastAPI Lab. Make sure to run train.py to get the model.')
                st.toast(':red[Model wine_model.pkl not found. Please run the train.py file in FastAPI Lab]', icon="🔥")
        else:
            LOGGER.error('Provide a valid JSON file with all 13 wine input features')
            st.toast(':red[Please upload a JSON test file. Check data folder for sample test file.]')

if __name__ == "__main__":
    run()
