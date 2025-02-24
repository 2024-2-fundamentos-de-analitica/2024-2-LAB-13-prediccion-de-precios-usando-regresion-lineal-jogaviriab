import pandas as pd
import gzip
import zipfile
import json
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score

def load_dataset_from_zip(zip_path):
    """Carga un dataset desde un archivo ZIP."""
    with zipfile.ZipFile(zip_path, "r") as z:
        file_name = z.namelist()[0]
        with z.open(file_name) as f:
            return pd.read_csv(f)

def preprocess_data(data):
    """Preprocesa los datos: crea la columna 'Age' y elimina columnas innecesarias."""
    data["Age"] = 2021 - data["Year"]
    data.drop(columns=["Year", "Car_Name"], inplace=True)
    return data

def build_regression_pipeline(x_train):
    """Construye un pipeline para el modelo de regresión."""
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = ["Selling_Price", "Driven_kms", "Owner", "Age"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('scaler', MinMaxScaler(), numerical_features),
        ]
    )
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)), 
        ('regressor', LinearRegression())
    ])
    
    return model_pipeline

def tune_hyperparameters(model, x_train, y_train):
    """Optimiza los hiperparámetros del pipeline usando validación cruzada."""
    param_grid = {"feature_selection__k": range(1, 12)}
    search = GridSearchCV(model, param_grid, n_jobs=-1, cv=10, scoring="neg_mean_absolute_error", refit=True)
    search.fit(x_train, y_train)
    return search

def save_trained_model(model, path="files/models/model.pkl.gz"):
    """Guarda el modelo entrenado en un archivo comprimido."""
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """Calcula métricas de desempeño del modelo."""
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    metrics_train = {
        "type": "metrics", "dataset": "train", 
        "r2": float(r2_score(y_train, y_train_pred)),
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "mad": float(median_absolute_error(y_train, y_train_pred))
    }
    
    metrics_test = {
        "type": "metrics", "dataset": "test", 
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "mad": float(median_absolute_error(y_test, y_test_pred))
    }
    
    return metrics_train, metrics_test

def save_metrics_to_file(metrics_train, metrics_test, file_path="files/output/metrics.json"):
    """Guarda las métricas en un archivo JSON."""
    with open(file_path, "w") as f:
        for metrics in [metrics_train, metrics_test]:
            f.write(json.dumps(metrics) + "\n")

# Cargar y preprocesar los datos
train_data = preprocess_data(load_dataset_from_zip("files/input/train_data.csv.zip"))
test_data = preprocess_data(load_dataset_from_zip("files/input/test_data.csv.zip"))

# Dividir en características (X) y variable objetivo (y)
x_train, y_train = train_data.drop(columns=["Present_Price"]), train_data["Present_Price"]
x_test, y_test = test_data.drop(columns=["Present_Price"]), test_data["Present_Price"]

# Construcción y optimización del modelo
regression_pipeline = build_regression_pipeline(x_train)
optimized_model = tune_hyperparameters(regression_pipeline, x_train, y_train)

# Guardar el modelo entrenado
save_trained_model(optimized_model)

# Evaluar el modelo y guardar métricas
evaluation_metrics_train, evaluation_metrics_test = evaluate_model(optimized_model, x_train, y_train, x_test, y_test)
save_metrics_to_file(evaluation_metrics_train, evaluation_metrics_test)
