import streamlit as st
from PIL import Image
from utils import predict_image, get_model_architecture
from logger import logger
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# ============ Page Config ==========
st.set_page_config(page_title="Mushroom", page_icon="M", layout="wide", initial_sidebar_state="auto")
st.title("Mushroom-Image-Classification")
st.caption("Mushroom Insight: Identify Before You Eat")
st.divider()

# ============ Sidebar ==============
st.sidebar.header("Image Classifier")
st.sidebar.divider()
screen = st.sidebar.radio("Navigate to :point_down:",options = ["Mushroom Insight", "Local Setup"])

# ========== Main Window ============
if screen == "Mushroom Insight":
    upload_status = False
    upload_column, result_column = st.columns([0.4, 0.6], gap="large")
    with upload_column:
        st.subheader("File Upload")
        image_file = st.file_uploader("Uplaod Image of Mushroom to identify it", 
                                        type=['png', 'jpg'],
                                        label_visibility="collapsed",)
        if image_file is not None:
            upload_status = True
            image_name = image_file.name
            image_uploaded = Image.open(image_file)
            display_image = image_uploaded.resize((256,256))
            st.toast("Image Uploaded Successfully!")
            st.divider()
            st.subheader("Image Uploaded") 
            st.image(display_image, caption='Uploaded Image. (Cropped for display)', use_column_width=False)
            
    with result_column:
        st.subheader("Prediction Results") 
        if upload_status:
            with st.spinner("Prediction in progress !"):
                # predictions = predict_image()
                prediction_response = predict_image(image_uploaded, image_name)
                if prediction_response.get("status") == "success":
                    prediction_results = prediction_response.get("response")
                    
                    mush_class = prediction_results["class"]
                    confidence = prediction_results["confidence"]
                    mush_type = prediction_results["class_type"]
                    softmax_proba = prediction_results["softmax"]
                    classes = prediction_results["classes"]

                    st.toast("Image Prediction completed")
                    st.write(f"Class: **{mush_class}**")
                    st.write(f"Mushroom-Type: **{mush_type}**")
                    st.write(f"Confidence: **{confidence:.2f}**")
                else:
                    st.error(prediction_response.get("response"))
                    st.stop()
                # Displaying Image and Chart
                chart_tab, image_tab = st.tabs(["Probability Distribution Chart", "Key Features"])
                with chart_tab:
                    chart_df = pd.DataFrame({"classes": classes, "Probability": softmax_proba})
                    axis = chart_df.plot.barh(label=classes)
                    y_positions = np.arange(len(classes))
                    plt.yticks(y_positions, classes)
                    st.pyplot(axis.figure)
                with image_tab:
                    image_path = "./Images/" + image_name
                    image_superimposed = Image.open(image_path)
                    image_superimposed = image_superimposed.resize((384,384))  
                    st.image(image_superimposed, caption='Key Features', use_column_width=False)

    st.divider()
else:
    st.subheader("Local Setup")

    st.write("**Clone Repository**")
    st.caption("Clone the git repo using the following command to your local directory")
    st.code("git clone https://github.com/DhaneshGavimath/Mushroom-Image-Classifier.git\ncd Mushroom-Image-Classifier",
            line_numbers=True)

    st.write("**Environment Setup**")
    st.caption("Create python environment and activate it")
    st.code("python -m venv .env\n.env\Scripts\\activate", line_numbers=True)
    st.caption("Install the requirements")
    st.code("pip install -r requirements.txt", line_numbers=True)

    st.write("**Start the app**")
    st.caption("Start the streamlit application with the following command. Provide the port number of your choice")
    st.code("streamlit run streamlit.py --server.port 9999", line_numbers=True)
    st.divider()


    