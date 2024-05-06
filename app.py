import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import csv
import pandas as pd
from streamlit_option_menu import option_menu
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64
# Load the saved CNN model
model = load_model('model/cnn_model.h5')

# Define the function to make predictions
def predict_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    if result[0][0] == 1:
        return 'dog'
    else:
        return 'cat'

def create_pdf_report(name, uploaded_file, prediction):
    buffer = BytesIO()
    c = canvas.Canvas(buffer,pagesize=letter)
    width,height = letter
    c.drawString(100, height-50, "Prediction Report")
    c.drawString(100, height-60, "--------------------------------------------")
    c.drawString(100, height-90, f"Name: {name}")
    
    c.drawString(100, height-120, "Uploaded Image")
    # image ta pdf e anchi
    if uploaded_file is not None:
        img = ImageReader(uploaded_file)
        c.drawImage(img, 100, 460, width=200, height=200)

    c.drawString(100, height-100, "--------------------------------------------")
    c.drawString(100, 420, f"Model Prediction: {prediction}")
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# Streamlit app
if __name__ == '__main__':
    st.markdown('## Cat or Dog Classifier \nUsing CNN')
    with st.sidebar:
        selected = option_menu('Cat-Dog Classifier',
                               ['Upload Image',
                                'Usage Records',
                                'More Info'],
                                icons=['upload','book','info'],
                                default_index=0)

    if selected =="Upload Image":
        name = st.text_input("**Your Name**")
        uploaded_file = st.file_uploader("**Choose a cat or dog image**", type=['jpg', 'png'])

        if st.button('Predict'):
            if uploaded_file is not None:
                 # Display image
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
                img = image.load_img(uploaded_file, target_size=(64, 64))

                # Make prediction
                prediction = predict_image(img)
                st.write(f"Prediction: {prediction}")
                pdf_bytes = create_pdf_report(name,uploaded_file, prediction)
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<a href="data:application/pdf;base64,{pdf_base64}" download="ImagePrediction_report_{name}.pdf">Download Report</a>'
                st.markdown(pdf_display, unsafe_allow_html=True)

                f = open("usage_record.txt","a")
                f.write("\n")
                new_data=str([name,prediction])
                leng = len(new_data)
                f.write(new_data[1:leng-1])
                f.close()

    if selected == "Usage Records":
        st.markdown("<h3 style='text-align: center;'>PREDICTION RECORDS OF OUR PREVIOUS USERS</h1>", unsafe_allow_html=True)
        f = pd.read_csv("usage_record.txt")
        #st.table(f)
        st.table(f.style.set_table_attributes('style="width:100%;"'))
        st.markdown("____")
        st.write("All the records are stored only for academic and research purpose & will not be used for any other means.")
    
    if selected == "More Info":
        st.markdown("<h2 style='text-align: center;'>ABOUT</h2>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<p style='text-align: center;'>This is an academic project made by B.Tech Computer Science And Engineering 3rd year student.</p>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<h4 style='text-align: center;'>Developed and maintained by</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Sankalpa Pramanik</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><b>psankalpa2019@gmail.com</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Ritabrata Dey</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><b>ritabratadey1296@gmail.com</b></p>", unsafe_allow_html=True)
        st.markdown("____")