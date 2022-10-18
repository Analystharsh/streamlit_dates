import streamlit as st
import tensorflow as tf
import streamlit as st
import gdown

@st.cache(allow_output_mutation=True)
def load_model():
  url = 'https://drive.google.com/file/d/1bl0GUexsmr_NLWQ2SM7HeRbmzk_ak_CL/view?usp=sharing'
  output = 'my_model2.hdf5'
  gdown.download(url, output, quiet=False)
  model=tf.keras.models.load_model('my_model2.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Dates Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
class_names = ['bumaan', 'fardh', 'khalas', 'lulu', 'shishi']
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (124,124)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    st.write(np.argmax(score))
    st.write(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
    #print(
    #"This image most likely belongs to {} with a {:.2f} percent confidence."
    #.format(class_names[np.argmax(score)], 100 * np.max(score))
#)

