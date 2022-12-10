import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
from keras.models import load_model

st.title('Taller 4,  GANS')

url = 'https://github.com/DavidPachis/DeepLearning/raw/main/taller4/models/Gen1-0.h5'
response = requests.get(url)
open("Gen1-0.h5", "wb").write(response.content)
best_model = load_model('model.h5')


def generate():
    noise = np.random.normal(loc=0, scale=1, size=(100, 100))
    gen_image = best_model.predict(noise)
    fig, axe = plt.subplots(2, 5)
    fig.suptitle('Generated Images from Noise using DCGANs')
    idx = 0
    for i in range(2):
        for j in range(5):
            axe[i, j].imshow(gen_image[idx].reshape(28, 28), cmap='gray')
            idx += 3


# Boton para disparar los generadores
if st.button('Make Prediction'):
    generate()
    st.write("Thank you! I hope you liked it.")
