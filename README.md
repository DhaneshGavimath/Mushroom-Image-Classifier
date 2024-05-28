## Mushroom-Image-Classifier
Training and deploying CNN to identify varieties of mushrooms which helps identify the edible and non-edible ones. Knowing the type of mushrooms is life saviour.

## Try the app - [Mushroom Image Classifier](https://mushroom-image-classifier.streamlit.app/)
<p>
    <img src="notebook/mushrooms.png" style="width:100%; height:auto;" />
</p>

Image credit: [phienix_han](https://unsplash.com/@phienix_han)

## Local Setup
**Clone Repository**
<br>
>Clone the git repo using the following command to your local directory
```
git clone https://github.com/DhaneshGavimath/Mushroom-Image-Classifier.git
cd Mushroom-Image-Classifier
```

**Environment Setup**
<br>
>Create python environment and activate it
```
python -m venv .env
.env\Scripts\activate
```

>Install the requirements
```
pip install -r requirements.txt
```

**Start the app**
<br>
>Start the streamlit application with the following command. Provide the port number of your choice
```
streamlit run streamlit.py --server.port 9999
```

## Dataset Credits:
Arun K Soman - https://www.kaggle.com/datasets/chipprogrammer/mashroom-image-classification
