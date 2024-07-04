import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ESRGAN import ESRGAN
from torchvision import transforms
import torch

def load_model():
    model = ESRGAN()
    model.load_state_dict(torch.load('ESRGAN_generator_epoch_50.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

esrgan = load_model()

def preprocess(img):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(192)
    ])
    image_p = transform(img)
    image_p = image_p.unsqueeze(0) # Converting shape of tensor from [1, 28, 28] to [1, 1, 28, 28]
    return image_p

def make_prediction(image):
    predictions = []
    print("image", image.shape)
    outputs = esrgan(image)
    opimg = outputs[0].permute(1, 2, 0).cpu().detach().numpy()
    return opimg

st.title('Image Super Resolution')
st.text('ESRGAN')
img_upload = st.file_uploader(label="Upload image Here:", type=["png","jpg","jpeg"])


if img_upload:
    img = Image.open(img_upload)
    tensor_image = preprocess(img)

    prediction = make_prediction(tensor_image)


    st.header("Original Image")
    fig, ax = plt.subplots(figsize=(20, 20))
    original_image = tensor_image.squeeze().cpu().numpy()
    ax.imshow(np.transpose(original_image, (1, 2, 0)))
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)

    st.header("Super-Res Image")
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(prediction, cmap='gray')
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)
