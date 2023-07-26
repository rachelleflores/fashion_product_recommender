## LIBRARIES
import streamlit as st
import numpy as np
import pandas as pd
import math

from PIL import Image

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

## LOAD DATA
all = pd.read_csv(r'./../data/cleaned_data/filtered_all4.csv')
labels = pd.read_csv(r'./../data/cleaned_data/labels.csv')
labels = pd.Series(labels.category) # transform labels df to a serie
scaled_embeddings = np.load('scaled_embeddings.npy') 

## MODEL FOR IMAGE EMBEDDING
# Load the VGG16 model with pre-trained weights (excluding the fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False)

# Create an intermediate layer model to obtain embeddings from the fully connected layer
intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    
def main():
    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")
    
    st.title("Product Recommender using Image Similarity")
    st.write("Upload an image to find similar or complementary products")
    
    ## FUNCTIONS
    def load_image(image):
        return Image.open(image)
    
    def get_image_embedding(image):
        image = image.convert("RGB")  # Convert the image to RGB format if it has an alpha channel (4 channels)
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        image_norm = image.astype('float32') / 255.0
        embed_image = intermediate_layer_model.predict(image_norm)
        embedded_matrix = np.array([image.flatten() for image in embed_image])
        #scaled = StandardScaler().fit_transform(embedded_matrix)
        return embedded_matrix
    
    def get_similar_products(embedding1, embedding2):
        similarities = cosine_similarity(embedding1, embedding2)
    
        # Sort the product scores in descending order
        most_similar = (np.argsort(similarities[0])[::-1][:5]).tolist()
        #np.argsort(similarities[0])[::-1][:num_similar]
        sorted_scores = embedding2[most_similar]
        
        # Retrieve the top similar products and their scores
        similar_products = embedding2[most_similar]
        similar_scores = similarities[0][most_similar]
        
        return most_similar
    
    def complete_outfit(embedding1, embedding2, labels, num_similar=8):
        # Get the similarity scores of the given product with all other products
        similarities = cosine_similarity(embedding1, embedding2)
    
        # Sort the product scores in descending order
        most_similar = np.argsort(similarities[0])[::-1]
    
        # Dictionary to store selected products per category
        selected_products = {}
    
        # List to store the final selected similar products
        similar_products = []
    
        # Counter to keep track of how many items have been selected for each label
        label_counter = {label: 0 for label in range(8)}
    
        for index in most_similar:
            category = labels[index]
            if label_counter[category] < 1:
                similar_products.append(index)
                selected_products[category] = index
                label_counter[category] += 1
    
            if len(similar_products) >= int(num_similar):
                break
    
        return similar_products
    
    def prod_details(index_list):
        """This function will show the product details which would be prodided in the streamlit app"""

        # empty lists to store infos
        prods = []
        brands = []
        prices = []
        links = []
        images = []

        for index in range(0, len(index_list)):

                product_name = all['product'][index_list[index]]
                prods.append(product_name)

                brand =  all['brand'][index_list[index]]
                brands.append(brand)

                price = all['orig_price'][index_list[index]]
                prices.append(price)

                link = all['link'][index_list[index]]
                links.append(link)

                image = all['img'][index_list[index]]
                images.append(image)

         # thank you chatgpt for making my code work!😅 I was orignally only able to display 1 row of 3 items
        items_per_row = 3  # Number of items to display per row
        num_rows = (len(similar_products) + items_per_row - 1) // items_per_row

        for row in range(num_rows):
            cols = st.columns(items_per_row)
            for idx, col in zip(range(row * items_per_row, min((row + 1) * items_per_row, len(similar_products))), cols):
                with col:
                    po, b, pi, l, i = prods[idx], brands[idx], prices[idx], links[idx], images[idx]
                    st.image(i, use_column_width=True)  # Display the product image
                    st.write(f"**Product Brand:** {b}")
                    st.write(f"**Product Name:** {po}")
                    st.write(f"**Price:** €{pi}")
                    st.write(f"**Link:** [{po}](https://www.smallable.com{l})")
                    st.write("")  # Add an empty line for spacing between products
                    idx += 1
                    
    c1, c2 = st.columns((1/3, 2/3))
    with c1:
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    with c2:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.write("")
            st.write("")
            if st.button("Find Similar Products"):
                uploaded_image = Image.open(uploaded_image)
                uploaded_image_embedding = get_image_embedding(uploaded_image)
                # Find similar products
                similar_products = get_similar_products(uploaded_image_embedding, scaled_embeddings)
                with c2: 
                    st.subheader("Similar Products:")  
                    prod_details(similar_products)
                    
        with sc2:
            st.write("")
            st.write("")
            if st.button("Complete Your Attire"):
                uploaded_image = Image.open(uploaded_image)
                uploaded_image_embedding = get_image_embedding(uploaded_image)
                # Find similar products
                similar_products = complete_outfit(uploaded_image_embedding, scaled_embeddings, labels)
                with c2:
                    st.subheader("Complementary Products:")
                    prod_details(similar_products)
                                    
if __name__ == "__main__":
    main()