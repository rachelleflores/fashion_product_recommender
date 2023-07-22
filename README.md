# fashion_product_recommender
Final Project for the Ironhack Data Analytics Bootcamp

## OBJECTIVE:
This project aims to create a product recommender using image similarities. 

## PROCESS:
To do so, here are the steps taken:
1. **Planning**
   - Here is a link to the Notion 
1. **Gather Data:**
    - Web scraping has been performed to obtain men and women's product data from the smallable website:
          a. Product Brand
          b. Brief Product Description
          c. Price
          d. Greenable (certifies that this item has been made from environmentally friendly materials and ingredients, and contains no chemical substances)
          e. Product Image
          f. Link to individual products
          g. Category the Product Belongs to
          h. Product Color

2. **Data Cleaning**
3. **Querying**
     - Using SQL Workbench, perform some SQL queries on gathered and cleaned dataset
4. **Visualization**
     - Use Tableau to visualize some relationships in terms of average price, brands, and categories
6. **Image Pre-processing**
   - Transform the images into an array
   - Apply image embedding through transfer learning of VGG16 Model
   - Use cosine similarity on the embeddings to calculate similarities between images
7. **Product Recommender**
8. **Streamlit App**
9. **Presentation**
