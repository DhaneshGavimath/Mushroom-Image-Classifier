import tensorflow as tf
import numpy as np
from tensorflow import keras
import json
from logger import logger


def load_model(model_path):
    """
    Loading the model
    params:
        model_path: path to .keras model
    returns: tensorflow.keras.Model
    """
    logger.info("Loading Keras Model ...")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
    except Exception as exp:
        model = None
    finally:
        return model

def get_model_architecture():
    """
    provides get the architecture image of the model
    returns: image file path(png)
    """
    model_path = "./notebook/mushroom_classifier.keras"
    image_file_path = "./notebook/mushroom_classifier.png"
    response = {"status": "success", "response": "response"}
    try:
        logger.info("Fetching model Architecture Image")
        model = model = load_model(model_path)
        if model is None:
            raise Exception("Failed to load the model")
        tf.keras.utils.plot_model(model, to_file=image_file_path,
                               show_shapes=False,show_dtype=False,
                               show_layer_names=False,rankdir="TB",
                               expand_nested=False,dpi=200,
                               show_layer_activations=False,
                               show_trainable=False)
        response["status"] = "success"
        response["response"] = {"path": image_file_path}
    except Exception as exp:
        logger.exception("Loading Architecture Failed. Error:{}".format(str(exp)))
        response["status"] = "failed"
        response["response"] = str(exp)
    finally:
        return response

def image_to_tensor(image):
    """
    Converts PIL image to numpy array
    params:
        image: PIL image
    returns: tensorflow.Tensor
    """
    logger.info("Converting Image to tensor...")
    resized_image = image.resize((128,128))
    numpy_array = np.array(resized_image)
    image_tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
    logger.info("Converting Image to tensor completed. ")
    return image_tensor

def predict_image(image_uploaded):
    """
    1. Loading the .keras model
    2. predicting the mushroom type
    params:
        image: tensor
    returns: dict -> {"class": "abc", "confidence":0.8, "class_type": "edible"}
    """
    response = {"status": "success", "response": "response"}
    model_path = "./notebook/mushroom_classifier.keras"
    classes_list_path = "./notebook/classes.json"

    try:
        logger.info("Invoked image prediction module ...")
        model = load_model(model_path)
        if model is None:
            raise Exception("Failed to load the model")
        with open(classes_list_path, "r") as file:
            classes_dict = json.load(file)
            classes = classes_dict.get("classes")
            
        # prcessing image
        image_tensor = image_to_tensor(image_uploaded)

        # 1. Adding Batch dimention
        input_image = tf.expand_dims(image_tensor, 0)
        # 2. Image scaling
        input_image = input_image * (1./255)

        prediction = model.predict(input_image)
        prediction_softmax = tf.nn.softmax(prediction, axis=None)

        # Prediction Results
        confidence_score = tf.math.reduce_max(prediction_softmax[0], axis=-1, keepdims=False).numpy()
        class_index = tf.math.argmax(prediction_softmax[0], axis=-1, output_type=tf.dtypes.int64)
        prediction = classes[class_index]
        prediction_class = prediction.split("-")[0]
        prediction_class_type = prediction.split("-")[1]
        prediction_results = {"class": prediction_class, 
                            "confidence":confidence_score, 
                            "class_type": prediction_class_type,
                            "softmax": prediction_softmax[0],
                            "classes": classes}
        
        response["status"] = "success"
        response["response"] = prediction_results
        logger.info("Image prediction successful.")
    except Exception as exp:
        logger.exception("Image prediction Failed. Error:{}".format(str(exp)))
        response["status"] = "failed"
        response["response"] = str(exp)
    finally:
        return response
