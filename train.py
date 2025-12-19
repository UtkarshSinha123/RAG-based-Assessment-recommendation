import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib  # or use `import pickle`

# Step 1: Load training data
df = pd.read_csv('train.csv')  # Make sure train.csv is in your working directory
X = df['query']
y = df['intent']  # or the correct label column name

# Step 2: Create a pipeline for text vectorization and classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Step 3: Train the model
pipeline.fit(X, y)

# Step 4: Save the model to disk using joblib (or use pickle)
joblib.dump(pipeline, 'intent_model.pkl')
