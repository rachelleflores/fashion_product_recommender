# fashion_product_recommender
Final Project for the Ironhack Data Analytics Bootcamp

## OBJECTIVE:
This project aims to create a product recommender using image similarities. 

## PROCESS:
To do so, here are the steps taken:
1. **Planning**
   - Here is a link to the [Notion Task List](https://rachellef.notion.site/865707cd5fee401b9abb1c2fe50b8a1d?v=770abdde733044afba0b426f550495be)
1. **Gather Data:**
    - Web scraping has been performed to obtain men's and women's product data from the smallable website:
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
     - Use [Tableau](https://public.tableau.com/views/fashion_product_distribution_twb/Dashboard22?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link) to visualize some relationships in terms of price ranges, brands, and categories
6. **Image Pre-processing**
   - Transform the images into an array
   - Apply transfer learning with the VGG16 Model to get image embeddings
   - Use cosine similarity on the embeddings to calculate similarities between images
7. **Product Recommender**
8. **Streamlit App**
9. **Presentation**
