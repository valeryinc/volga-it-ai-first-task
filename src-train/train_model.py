import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import chardet

def get_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_data(file_path):
    try:
        file_encoding = get_encoding(file_path)
        return pd.read_csv(file_path, delimiter=';', encoding=file_encoding)
    except Exception as e:
        print(f"Error while loading the data: {e}")
        return None

def preprocess_data(data):
    data.dropna(subset=['Desc', 'Group', 'Cat'], inplace=True)
    data['Desc'] = data['Desc'].str.lower()
    return data

def train_model(data, model_path):
    data = data.dropna(subset=['Desc'])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Desc'])
    y = data['Cat']
    joblib.dump(vectorizer, "../model/vectorizer.pkl")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Здесь, я предполагаю, что вы хотите использовать RandomForestClassifier
    # Сделайте корректировки при необходимости
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Выведите отчет о классификации для оценки
    print(classification_report(y_test, y_pred))

    # Save the trained model
    try:
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        joblib.dump(model, model_path)
    except Exception as e:
        print(f"Error while saving the model: {e}")

if __name__ == "__main__":
    data = load_data("../L.csv")
    if data is not None:
        train_model(data, "../model/model.pkl")
