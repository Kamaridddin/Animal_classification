import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


#title
st.title('Animal finder model')

#upload
file = st.file_uploader('upload photo', type=['png', 'jpeg', 'gif', 'svg', 'jpg'])
if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)
    #model
    model = load_learner('animal_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f'Probablity: {probs[pred_id]*100:.1f}')

    #plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)