import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_and_predict(csv_path):
    df = pd.read_csv(csv_path)

    # Example columns: ['Attendance', 'Marks', 'Assignments', 'Passed']
    X = df[['Attendance', 'Marks', 'Assignments']]
    y = df['Passed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save trained model
    joblib.dump(model, 'predictor/student_model.pkl')

    return accuracy, df.shape[0]
