# Fortune Medical Associates (Predicting Drug Review Results)
**By Yamuna Umapathy**

<p align="center">
    <img src = "https://github.com/YamunaU75/FortuneMedical/blob/main/Data/TopPicture1.jpg" width = "700" height="204">
</p>


## Business Understanding:

Our Stakeholder Fortune Medical Associates wants us to classify Positive & Negative ratings from given Patient's Drug Review results. We have to
find the best model which predicts patient's review has positive or negative effect after medication.

## Dataset:

Dataset comes from UCI ML Respository: <a href = "https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com">.

The dataset provides patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient 
satisfaction. Dataset was cleaned, and has 215,000 rows and 6 features. Two important features used in our analysis, description as below:

`review`(text): Patient review after taking drug (categorical).

`rating`      : Patient's review rating after taking medication(float)

## Data Exploration & Cleaning:

Dataset has 215K rows and 6 columns, dropped rows with missing values. Encoded column `rating` as 1 for ratings 6-10 and 0 for ratings 1-5. 
Visualizing positive review words and negative review words after preprocessing through WordCloud.

<p align="center">
    <img src = "https://github.com/YamunaU75/FortuneMedical/blob/main/Data/Pos_WordCl2.jpeg" width = "900" height="675">
</p>

## Preprocessing:

Using TextPreprocessor class to normalize our text, following cleaning was done: Lowercasing, Tokenizing and removing stop words, removing special
characters and numbers, tagging parts of speech, Lemmatizing or Stemming process.

Since preprocessing steps for 215K rows consumed long hours, I chose a sample Dataset of 100K, balanced dataset choosing rating 1 and 0 equally, 
solved class imbalance issue. 

## Baseline Model:

After preprocessing the input text data `review`, used pipeline to complete vectorizing steps and run baseline models. Baseling model with Random
Forest with parameters n_estimators=100, max_depth=3 scored initial accuracy score of 65.5. Since dataset is balanced one, using Accuracy
as Metrics.

## Randomized SearchCV:

After running Baseline Model with Random Forest, I used tuned hyperparameters through Randomized SearchCV to run next models. Since GridSearchCV
consumes time consuming and less cost efficiency, chose Randomized SearchCV and ran with Scikit Learn Random Forest & XGBoost models. 

Random Forest with RCV option scored the best Accuracy score 76, and XGBoost with RCV option scored 71 as Accuracy score. I got next better model
as Random Forest with RCV option than our Baseline model, checking TensorFlow next.

## TensorFlow:

Inputting the dense array of preprocessed column 'review' to TensorFlow Model Sequential(), and choose ('Binary cross entropy' as category, metric=
'accuracy', epochs = 10, validation_size = 0.2 and Early stopping) to run first model with Tensorflow. Results showed best training scores and lower 
test scores which clearly showed model was 'overfitting'. By applying Regularization l2, TensorFlow model resulted next best accuracy score 78.89, 
and Training loss graph was slightly away from validation accuracy due to regularization applied.

<p align="center">
    <img src = "https://github.com/YamunaU75/FortuneMedical/blob/main/Data/training_loss_history_tensor.jpeg" width = "600" height="451">
</p>

## Stacking:

Stacking is a learning technique that combines all Base models predictions, and meta-model takes the predictions from the base models as input features and learns how to weigh or combine them effectively. I carefully selected 3 Base models based on following reasons to be best suited for text data.

Linear Support Vector Classifier as first model, works often faster and more efficient when dealing with large datasets. Linear SVC is suitable for high-dimensional and sparse data, which is often the case with text data. 

Using Naive Bayes as second model, which is probabilistic classifier that assumes independence between features (words) given the class. It is known for its simplicity and efficiency and can work well with text data (Input). Naive Bayes classifiers are based on the Bayes' theorem and make predictions by calculating the probability of a class given the observed features. They can handle high-dimensional data efficiently, making them suitable for text classification tasks.

Logistic Regression as third model, another popular choice for binary classification with text data. It models the relationship between the features and the log-odds of the target class, allows to estimate the probability of the target class.
 
Meta model: Logistic Regression. Stacking outperformed all other models, and scored **best Accuracy score 80.3**.

**Confusion Matrix for Stacking Model:**
<p align="center">
    <img src = "https://github.com/YamunaU75/FortuneMedical/blob/main/Data/confusionmatrix_stmodel.jpeg" width = "600" height="451">
</p>

False Positive: Negative Reviews which was assigned under Positive reviews, these patients needs immediate follow up appointment to diagnose why Drug
failed to work.

False Negative: Positive reviews assigned under Negative reviews, means these patients are doing well after medication, and contact doctor office
if they have any questions. 

**Visualizing Feature Importance**
<p align="center">
    <img src = "https://github.com/YamunaU75/FortuneMedical/blob/main/Data/Top15Feature_Stmodel.jpeg" width = "600" height="451">
</p>

## Topic Modeling:

Topic modeling is a statistical modeling technique used to identify latent topics or themes within a collection of documents. It is an unsupervised learning method that aims to discover the underlying patterns and structure in a text corpus.

Non-Negative Matrix Factorization (NMF) is an alternative technique for topic modeling, which is based on linear algebra and matrix factorization. NMF is often used for dimensionality reduction and feature extraction tasks. The algorithm factorizes the document-term matrix into the document-topic matrix and the topic-term matrix. The number of topics to extract needs to be specified beforehand.

$ X = WH $
- $ W $ encodes the importance of each token in the fitted topics. 
- $ H $ encodes the weight of the fitted topics for each document.

I chose n-components as 2 for two topics, and ran NMF topic model which derived words belonging to positive and negative reviews. Also, using python library
pyLDAVis, visualization clearly showed words belonging to 2 different topics, and saved as nmf.html file.

html link: <a href = "C:/Users/uyamu/Documents/FortuneMedical/nmf_topics.html">

<p align="center">
    <img src = "https://github.com/YamunaU75/FortuneMedical/blob/main/Data/Screenshot%202024-04-10%20122228.jpg" width = "600" height="451">
</p>

## Conclusion:

Based on model evaluation, Stacking Model using base models (Linear SVC, Naive Bayes & Logistic Regression) performed the best accuracy score 80.3 compared to others. We are focussing on metrics Accuracy since our dataset is balanced, and sample was taken from the main dataset. 

Confusion matrix and Classification report implies that our Stakeholder can focus on True Negative and False Positive, means patient's Drug effect was Negative, these patients needs immediate follow up to diagnose why Drug failed to work.

Finally, We also looked into Topic Modeling, this was very helpful to distinguish between Posive and Negative ratings clearly. Topic Modeling helped us to visualize topics through pyLDAvis Visualization.

**Next Steps:**

For Phase 2, we can look into using Word2Vec algorithm, widely used word representation technique. This type of word representation or Word Embeddings can be used to infer semantic similarity between words and phrases, expand queries, surface related concepts and more. Also how to correct spelling mistakes before running model using TextBlob or PySpellCheck python libraries.

## Repository Structure

```
├── Data
├── .gitignore
├── FortuneMedicalMain.ipynb
├── README.md
├── TensorFlow.ipynb
└── nmf.html
└── FortuneMedicalPresentation.pdf
```



