from PIL import Image
import numpy as np

import yaml
import streamlit as st

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['frontend']


def grayscale_image(image):
    grayscale_array = np.array(image.convert('L'))
    return Image.fromarray(grayscale_array)

def main():
    st.title("Simple Image Grayscale Converter")
    st.write(f"Host: {config['host']}, Port: {config['port']}")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_column_width=True)

        if st.button("Convert to Grayscale"):
            grayscale_img = grayscale_image(original_image)
            st.image(grayscale_img, caption="Grayscale Image", use_column_width=True)

if __name__ == "__main__":
    main()
