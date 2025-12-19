import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

# Load test queries
df_test = pd.read_csv('Gen_AI Dataset.xlsx', sheet_name='test')  # Adjust sheet_name if needed
queries = df_test['query']

# Predict
predictions = model.predict(queries)

# Save to CSV in required format
output = pd.DataFrame({'query': queries, 'predictions': predictions})
output.to_csv('predictions.csv', index=False)
