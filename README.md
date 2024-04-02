# Image Captioning with Pytorch

This project is an image captioning model using computer vision and natural language processing. The deep learning is an encoder-decoder CNN+LSTM model with Bahdanau Attention. The encoder uses effientnet v2 medium as feature extraction from the images and LSTM for caption generation using the extracted features.

The model was trained on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) Dataset for image captioning. We also used the [Flickr](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) Image captioning dataset to increase the vocabulary size to enable the model to perform better on new images.

## Requirements
Install the packages needed to run the code in a python environment.
```
pip install requirements.txt
```

## Example
The code can be ran using the command below.
```
python getCaption.py -i ./images/IMG-20220617-WA0058.jpg
```
The results of the example is given below.
![Example](https://github.com/Kokotla/image_captioning/blob/main/res/sample_image.png)

