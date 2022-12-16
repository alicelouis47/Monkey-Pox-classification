import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from torch import topk
import skimage.transform
import matplotlib.pyplot as plt

from PIL import Image
import streamlit as st
import urllib.request
import os

from matplotlib.pyplot import imshow

with st.sidebar:

    st.header("🖥️About Project")
    st.write("The developer is interested in developing a model to classify monkeypox. This project studied the classification of images of monkeypox into 2 categories: 1) images of monkeypox and 2) images of non-monkeypox. Based on the image dataset that the developers have collected from reliable sources on the Internet, three pre-trained models, ConvNeXt_Small, RegNet_Y_16GF, and Wide_ResNet50-2, were used to optimize the pre-trained model with the best performance parameters. Model to fit the set of images. and evaluate model performance with accuracy, precision, recall, and F1-score.",unsafe_allow_html=True)
    
    st.header("🌐reference")
    st.write("Monkeypox Image Data collection.[more](https://arxiv.org/abs/2206.01774)")

st.header('Monkey Pox classification: จำแนกโรคฝีดาษลิง🙈🙉🙊: サル痘の分類')


with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
hide_table_index = """
            <style>         
            thead {display:none}  
            tbody th {display:none}
            .blank {display:none}
            </style>
            """ 
st.markdown(hide_table_index, unsafe_allow_html=True)


#download model
model_url = "https://huggingface.co/alicelouis/MonkeyPoxClassification/resolve/main/AdamW_MSD_REGNET_Y_16GF.pt"
urllib.request.urlretrieve(model_url,"AdamW_MSD_REGNET_Y_16GF.pt")

#load backbone
model = models.regnet_y_16gf(pretrained=True)
#change fc 1000 class --> 2 class
for param in model.trunk_output[0:].parameters():
    param.requires_grad = False
model.fc = nn.Linear(in_features=3024, out_features=2)
# use cpu tp processing
device = torch.device('cpu')

model.load_state_dict(torch.load('AdamW_MSD_REGNET_Y_16GF.pt', map_location=device))

uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_out = img
    img_out = np.array(img_out)
    # load save weight model
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])
    input = transform(img)
    heat_map = transform(img)
    prediction_var = Variable((input.unsqueeze(0)).cpu(), requires_grad=True)
    model.eval()
    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    def predict_classes(pred_probabilities):
        if pred_probabilities[0] >= 0.7:
            return "Monkeypox"
        else:
            return "Non-Monkeypox"
    #Heat Map
    preprocess = transforms.Compose([transforms.Resize((224,224)),
   transforms.ToTensor(),])

    display_transform = transforms.Compose([transforms.Resize((224,224))])

    tensor = preprocess(img)

    prediction_var = Variable((tensor.unsqueeze(0)).cpu(), requires_grad=True)

    class SaveFeatures():
        trunk_output=None
        def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
        def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
        def remove(self): self.hook.remove()

    final_layer = model.trunk_output.block4

    activated_features = SaveFeatures(final_layer)

    prediction = model(prediction_var)

    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()

    topk(pred_probabilities,1)

    def getCAM(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    
    class_idx = topk(pred_probabilities,1)[1].int()

    overlay = getCAM(activated_features.features, weight_softmax, class_idx )
    
    imshow(display_transform(img))
    imshow(skimage.transform.resize(overlay[0], input.shape[1:3]), alpha=0.5, cmap='jet')
    
    plt.savefig('img_HeatMap.png', bbox_inches='tight', pad_inches=0)

    img_HeatMap = Image.open('img_HeatMap.png')

    st.success(f"This is {predict_classes(pred_probabilities)}  with the probability of {pred_probabilities[0]*100:.02f}%")

    st.image([img_out,img_HeatMap],width=350)
    os.remove('img_HeatMap.png')
