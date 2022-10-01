import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np

from PIL import Image
import streamlit as st
import urllib.request



st.title('Monkey classification')

# #download model
# model_url = "https://huggingface.co/alicelouis/MonkeyPoxClassification/resolve/main/AdamW_MSD_REGNET_Y_16GF.pt"
# urllib.request.urlretrieve(model_url,"AdamW_MSD_REGNET_Y_16GF.pt")

#load backbone
model = models.regnet_y_16gf(pretrained=True)
#change fc 1000 class --> 2 class
for param in model.trunk_output[0:].parameters():
    param.requires_grad = False
model.fc = nn.Linear(in_features=3024, out_features=2)
# ใช้ CPU ประมวลผล
device = torch.device('cpu')

model.load_state_dict(torch.load('AdamW_MSD_REGNET_Y_16GF.pt', map_location=device))



uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")






if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_out = img
    img_out = np.array(img_out)
    # โหลดโมเดลที่เซฟ
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])
    input = transform(img)
    prediction_var = Variable((input.unsqueeze(0)).cpu(), requires_grad=True)
    model.eval()
    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    def predict_classes(pred_probabilities):
        if pred_probabilities[0] >= 0.7:
            return "Monkeypox"
        else:
            return "Non-Monkeypox"
    st.success(f"This is {predict_classes(pred_probabilities)}  with the probability of {pred_probabilities[0]*100:.02f}%")
    st.image(img_out, use_column_width=True)
