<h1>Text Sentiment Analysis</h1>

<h3>Overview</h3>
This repository contains code for a sentiment analysis model trained on text data. The model predicts whether a given text sentiment is positive, negative, or neutral. It utilizes the AdaBoostClassifier algorithm with a DecisionTreeClassifier base estimator.

<h3>Dataset </h3>
The model was trained on a dataset obtained from Kaggle which is also available in this repository, which consists of text data labeled with sentiment categories. The original dataset contained the following columns:
<ul>
<li><b>`tweet_id`</b> : Unique identifier for each tweet</li>
<li><b>`sentiment`</b> : Categorical sentiment label (e.g., happiness, sadness, anger, etc.)</li>
<li><b>`content`</b> : Textual content of the tweet </li>  
</ul>

We processed this dataset to create the following columns:
<ul>
<li><b>`sentiment_category`</b> : Target sentiment category (positive, negative, neutral)</li>
<li><b>`neg_score`, `neu_score`, `pos_score`</b> : generated using NLTK sentiment analysis tools. </li>
</ul>

<h3>Preprocessing</h3>
<ul>
<li>Text data is preprocessed using TF-IDF vectorization.</li>
<li>Numeric features (<b>`neg_score`, `neu_score`, `pos_score`</b>) are scaled using StandardScaler.</li>
<li>Preprocessing is performed using scikit-learn's ColumnTransformer.</li>
</ul>

<h3>Model Pipeline</h3>
The model pipeline consists of the following steps:
<ol>
<li>Preprocessing: Text data is preprocessed using TF-IDF vectorization and numeric features are scaled.</li>
<li>Classification: AdaBoostClassifier with DecisionTreeClassifier base estimator is used for classification.</li>
</ol> 

<h3>Hyperparameter Tuning</h3>
Hyperparameters of the AdaBoostClassifier and DecisionTreeClassifier are tuned using GridSearchCV and RandomizedSearchCV to optimize performance.

<h3>Model Evaluation</h3>
The model is evaluated using cross-validation and tested on a holdout test set.
Evaluation metrics include accuracy, precision, recall, and F1-score.

<h3>Visualization</h3>
<ul>
<li>Decision tree visualization: Subtree rooted at a specified node can be visualized using the visualize_subtree function.</li>
</ul>

<h3>Usage</h3>
<ol>
<li>Install the required dependencies listed in requirements.txt. </li>
<li>Run the provided code to train the model and tune hyperparameters.</li>
<li>Use the trained model to predict sentiment on new text data.</li>
</ol>

<h3>Files</h3>
<ul>
<li><b>`model_training.ipynb`</b>: Jupyter notebook containing code for model training, evaluation, and visualization.</li>
<li><b>`README.md`</b>: Documentation providing an overview of the project.</li>
<li><b>`requirements.txt`</b>: List of dependencies required to run the code.</li>
</ul>

<h3>Author</h3>
<h4>Vajinder Kaur</h4>
