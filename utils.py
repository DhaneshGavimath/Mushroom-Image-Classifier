import tensorflow as tf
import numpy as np
from tensorflow import keras
import json
from logger import logger
import matplotlib as mlp
import os


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

def load_grad_model(model, layer_name):
    """
    Creating a Tensorflow model which returns layer as additional output
    params:
        model: tensorflow.keras.Model
        layer_name: name of the last convolution layer
    returns: tensorflow.keras.Model
    """
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    return grad_model

def preprocess_image(image_uploaded):
    """
    preprocessing the input image
    """

    # 1. Converting image object to tensor
    image_tensor = image_to_tensor(image_uploaded)
    # 2. Adding Batch dimention
    input_image = tf.expand_dims(image_tensor, 0)
    # 3. Image scaling
    input_image = input_image * (1./255)

    return input_image

def image_rescale(image):
    pass

def predict_image(image_uploaded, image_name):
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
            
        # preprocess image
        input_image = preprocess_image(image_uploaded)

        # load grad_model
        conv_layer_name = "feature_maps_4D"
        grad_model = load_grad_model(model,conv_layer_name)

        # gradient tape to compute effect of convolution output feature maps on classification
        with tf.GradientTape() as tape:
            conv_output, prediction = grad_model(input_image)
            prediction_softmax = tf.nn.softmax(prediction, axis=None)
            confidence_score = tf.math.reduce_max(prediction_softmax, axis=-1, keepdims=True)
        gradient = tape.gradient(confidence_score, conv_output)
        avg_gradient_across_feature_map = tf.reduce_mean(gradient, axis=(0, 1, 2))
        avg_gradients = avg_gradient_across_feature_map[..., tf.newaxis]
        # conv dim : 1, 12, 12, 128  & avg_gradients dim: 128, 1
        feature_maps = conv_output[0] @ avg_gradients # 12, 12, 128 * 128, 1 => 12, 12, 1
        feature_maps = tf.squeeze(feature_maps)
        feature_maps_normalized = tf.maximum(feature_maps, 0) / tf.math.reduce_max(feature_maps)
        feature_maps_array = 255 * feature_maps_normalized
        feature_maps_array = np.uint8(feature_maps_array)
        # colormap
        width, height = image_uploaded.size
        jet_map =  mlp.colormaps["jet"]
        jet_colors = jet_map(np.arange(256))[:, :3]
        feature_map_colored = jet_colors[feature_maps_array]
        feature_map_colored = tf.keras.utils.array_to_img(feature_map_colored)
        feature_map_colored = feature_map_colored.resize((width, height))
        feature_map_colored = keras.utils.img_to_array(feature_map_colored)
        # Image folder
        image_folder = os.path.join(os.getcwd(),"Images")
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        save_path = os.path.join(image_folder, image_name)
        # Superimposing the images
        transparancy = 0.3
        image_uploaded_array = np.array(image_uploaded)
        super_imposed_array = image_uploaded_array + transparancy * feature_map_colored
        super_imposed_image = tf.keras.utils.array_to_img(super_imposed_array)
        super_imposed_image.save(save_path)

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
