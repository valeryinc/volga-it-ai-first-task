import pandas as pd
import joblib
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer

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

def load_model(model_path):
    return joblib.load(model_path)

def load_vectorizer(vectorizer_path):
    return joblib.load(vectorizer_path)

def split_prediction(pred):
    parts = pred.split("_", maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return pred, None

def predict_new_data(data, model, vectorizer):
    vectorized_data = vectorizer.transform(data['Desc'].str.lower())
    data['Prediction'] = model.predict(vectorized_data)
    data['Group'], data['Cat'] = zip(*data['Prediction'].apply(split_prediction))
    data.drop(columns=['Prediction'], inplace=True)
    return data

def save_predictions(data, file_path):
    data.to_csv(file_path, sep=';', index=False)

if __name__ == "__main__":
    # Загружаем модель
    model = load_model("../model/model.pkl")
    
    # Загружаем векторизатор
    vectorizer = load_vectorizer("../model/vectorizer.pkl")
    
    # Загружаем новые данные для предсказания
    new_data = load_data("../C.csv")
    
    # Предсказываем категории для новых данных
    predicted_data = predict_new_data(new_data, model, vectorizer)
    
    # Сохраняем предсказанные данные
    save_predictions(predicted_data, "../result/A.csv")
    print("Predictions saved to result/A.csv")