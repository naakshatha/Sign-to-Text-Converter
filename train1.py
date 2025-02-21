import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('C:/Users/akshu/Desktop/data/data.pickle', 'rb'))

# Preprocess data for uniformity
max_length = max(len(row) for row in data_dict['data'])
processed_data = [
    np.pad(row, (0, max_length - len(row)), mode='constant') if len(row) < max_length else row[:max_length]
    for row in data_dict['data']
]

data = np.asarray(processed_data)
labels = np.asarray(data_dict['labels'])

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Model training
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predictions
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save model
with open('C:/Users/akshu/Desktop/data/model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
