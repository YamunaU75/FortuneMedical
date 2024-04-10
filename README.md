# Fortune Medical Associates (Predicting Drug Review Results)
**By Yamuna Umapathy**

## Business Understanding:

Our Stakeholder Fortune Medical Associates wants us to classify Positive & Negative ratings from given Patient's Drug Review results. We have to
find the best model which predicts patient's review has positive or negative effect after medication.

## Dataset:

Dataset comes from UCI Machine Learning Respository, https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com. The dataset provides 
patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction.

Dataset was cleaned, and has 215,000 rows and 2 important features to be used for our analysis. Features description as below:

`review`(text): Patient review after taking drug (categorical).

`rating`      : Patient's review rating after taking medication(float)

## Data Exploration & Cleaning:

Dataset has 215K rows and 6 columns, dropped rows with missing values. Encoded column `rating` as 1 for ratings 6-10 and 0 for ratings 1-5. 
Visualizing positive review words and negative review words after preprocessing through WordCloud.

