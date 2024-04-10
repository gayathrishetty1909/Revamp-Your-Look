import os
import streamlit as st
import cv2
from demo import Makeup_Transfer
from PIL import Image

def main():
    st.title('Revamp Your Look')
    
    uploaded_file1 = st.file_uploader("Upload Your Image ", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1_path = os.path.join("temp", uploaded_file1.name)
        with open(image1_path, "wb") as f1:
            f1.write(uploaded_file1.read())
            
        image2_path = os.path.join("temp2", uploaded_file2.name)
        with open(image2_path, "wb") as f2:
            f2.write(uploaded_file2.read())

        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)
        st.image(image1, caption= 'Input Image')
        st.image(image2, caption= 'Reference Image')
        
        if st.button('Transfer Makeup'):
            output_image = Makeup_Transfer(image1_path, "temp2")
            #output_image = Image.open(output_image)
            st.image(output_image, caption='Output Image', use_column_width=True)
    
if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")
    if not os.path.exists("temp2"):
        os.mkdir("temp2")
    main()