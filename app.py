import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ESPCNN import ESPCNN
from FSRCNN import FSRCNN
from SRCNN import SRCNN
from torchvision import transforms
import torch

def load_ESPCNN_model():
    model = ESPCNN(scale_factor=1, num_channels=3)
    model.load_state_dict(torch.load('max_psnr_espcn1720194079.7983537.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def load_FSRCNN_model():
    model = FSRCNN()
    model.load_state_dict(torch.load('max_pcnr_fsrcnn1720193748.7443247.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def load_SRCNN_model():
    model = SRCNN(num_channels=3)
    model.load_state_dict(torch.load('max_pcnr_srcnn1720167926.4699183.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

espcnn = load_ESPCNN_model()
fsrcnn = load_FSRCNN_model()
srcnn = load_SRCNN_model()

def preprocess(img):
    transform = transforms.Compose([
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor()
    ])
    image_p = transform(img)
    image_p = image_p.unsqueeze(0) # Converting shape of tensor from [1, 28, 28] to [1, 1, 28, 28]
    return image_p

def make_prediction(image, modelName = "SRCNN"):
    print("image", image.shape)
    if modelName == "SRCNN":
        outputs = srcnn(image)
    elif modelName == "ESPCNN":
        outputs = espcnn(image)
    elif modelName == "FSRCNN":
        outputs = fsrcnn(image)
    opimg = outputs[0].permute(1, 2, 0).cpu().detach().numpy()
    return opimg

st.title('Image Super Resolution')
st.text('ESPCNN | FSRCNN | SRCNN')
img_upload = st.file_uploader(label="Upload image Here:", type=["png","jpg","jpeg"])


if img_upload:
    img = Image.open(img_upload)
    tensor_image = preprocess(img)

    prediction1 = make_prediction(tensor_image, "FSRCNN")
    prediction2 = make_prediction(tensor_image, "ESPCNN")
    prediction3 = make_prediction(tensor_image, "SRCNN")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        fig, ax = plt.subplots(figsize=(20, 20))
        original_image = tensor_image.squeeze().cpu().numpy()
        ax.imshow(np.transpose(original_image, (1, 2, 0)))
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.header("Super-Res Image")
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(prediction1, cmap='gray')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        st.text('FSRCNN')
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(prediction2, cmap='gray')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        st.text('ESPCNN')
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(prediction3, cmap='gray')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        st.text('SRCNN')
