USE smallable;

# check data in the table
SELECT *
FROM `all`;

# check how many products are considered green vs not
SELECT greenable, COUNT(greenable) AS number_of_green_products
FROM `all`
GROUP BY greenable;

# check how many products are considered discounted vs not
SELECT discounted, COUNT(discounted) AS number_of_discounted_products
FROM `all`
GROUP BY discounted;

# what are the top 10 most expensive products per gender and collection

-- women's spring-summer collection
SELECT product, color, brand, orig_price
FROM `all`
WHERE collection = "ss" AND gender = "w"
ORDER BY orig_price DESC
LIMIT 10;

-- women's fall-winter collection
SELECT product, color, brand, orig_price
FROM `all`
WHERE collection = "fw" AND gender = "w"
ORDER BY orig_price DESC
LIMIT 10;

-- men's spring-summer collection
SELECT product, color, brand, orig_price
FROM `all`
WHERE collection = "ss" AND gender = "m"
ORDER BY orig_price DESC
LIMIT 10;

-- men's fall-winter collection
SELECT product, color, brand, orig_price
FROM `all`
WHERE collection = "fw" AND gender = "m"
ORDER BY orig_price DESC
LIMIT 10;

# what is the average original price , most expensive and least expensive prices for each categories
SELECT label, ROUND(AVG(orig_price),2) as avg_price_per_category, MAX(orig_price) as most_expensive, MIN(orig_price) as cheapest
FROM `all`
GROUP BY label
ORDER BY avg_price_per_category DESC;

# what is the highest discount rate applied?
SELECT CONCAT(ROUND(((orig_price-disc_price)/orig_price)*100,2),"%") AS highest_discount_rate
FROM `all`
ORDER BY highest_discount_rate DESC
LIMIT 1;