import streamlit as st
from PIL import Image
from utils import predict_image
from logger import logger

st.set_page_config(page_title="Mushroom", page_icon="M", layout="wide", initial_sidebar_state="auto")
st.title("Mushroom-Image-Classification")
st.caption("Welcome to the application!")

st.sidebar.write("Welcome to Image classification")

upload_status = False

upload_column, result_column = st.columns([0.6, 0.4], gap="large")
with upload_column:
    st.header("Input")
    image_file = st.file_uploader("Uplaod Image of Mushroom to identify it", 
                                    type=['png', 'jpg'],
                                    label_visibility="collapsed",)
    if image_file is not None:
        image_uploaded = Image.open(image_file)
        display_image = image_uploaded.resize((256,256))
        st.success("Image Uploaded Successfully!")
        st.image(display_image, caption='Uploaded Image.(Cropped for display)', use_column_width=True)
        upload_status = True

with result_column:
    st.header("Prediction Results") 
    if upload_status:
        with st.spinner("Prediction in progress !"):
            # predictions = predict_image()
            prediction_response = predict_image(image_uploaded)
            if prediction_response.get("status") == "success":
                prediction_results = prediction_response.get("response")
                st.write("Prediction:",prediction_results["class"])
                st.write("Confidence:",prediction_results["confidence"])
                st.write("Mushroom-Type:",prediction_results["class_type"])
            else:
                st.error(prediction_response.get("response"))
        

    