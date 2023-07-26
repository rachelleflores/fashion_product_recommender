# fashion_product_recommender
Final Project for the Ironhack Data Analytics Bootcamp

## OBJECTIVE:
This project aims to create a product recommender using image similarities. 

## PROCESS:
To do so, here are the steps taken:
1. **Planning**
   - Here is a link to the [Notion Task List](https://rachellef.notion.site/865707cd5fee401b9abb1c2fe50b8a1d?v=770abdde733044afba0b426f550495be)
2. **Gather Data:**
    - Web scraping has been performed to obtain men's and women's product data from the smallable website:
    - Product Brand
    - Brief Product Description
    - Price
    - Greenable (certifies that this item has been made from environmentally friendly materials and ingredients, and contains no chemical substances)
    - Product Image
    - Link to individual products
    - Category the Product Belongs to
    - Product Color

3. **Data Cleaning**
4. **Querying**
     - Using SQL Workbench, perform some SQL queries on gathered and cleaned dataset
5. **Visualization**
     - Use [Tableau](https://public.tableau.com/views/fashion_product_distribution_twb/Dashboard22?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link) to visualize some relationships in terms of price ranges, brands, and categories
6. **Image Pre-processing**
   - Transform the images into an array
   - Apply transfer learning with the VGG16 Model to get image embeddings
   - Use cosine similarity on the embeddings to calculate similarities between images
7. **Product Recommender**
8. [**Streamlit App**](https://rf-fashion-recommender.streamlit.app/)
9. **Presentation**
