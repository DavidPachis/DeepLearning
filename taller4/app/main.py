import streamlit as st
import numpy as np
import requests
from keras.models import load_model

st.title('Taller 4,  GANS')

url = 'https://github.com/DavidPachis/DeepLearning/raw/main/taller4/models/Gen1-0.h5'
response = requests.get(url)
open("Gen1-0.h5", "wb").write(response.content)
best_model = load_model('model.h5')


def generate():
    noise = np.random.normal(loc=0, scale=1, size=(100, 100))
    gen_image = best_model.predict(noise)
    fig, axe = st.pyplot.subplots(2, 5)
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
