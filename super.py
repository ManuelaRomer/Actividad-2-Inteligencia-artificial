
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SistemaTransporteInteligente:
    def __init__(self):
        self.modelo = None
        self.label_encoders = {}

    def generar_dataset_simulado(self, n=200):
        data = {
            "origen": np.random.choice(["A", "B", "C", "D", "E"], n),
            "destino": np.random.choice(["B", "C", "D", "E", "F"], n),
            "hora_dia": np.random.randint(5, 22, n),
            "dia_semana": np.random.choice(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"], n),
            "distancia_km": np.round(np.random.uniform(2, 15, n), 2),
            "congestion": np.random.choice(["Baja", "Media", "Alta"], n),
            "duracion_min": np.round(np.random.uniform(10, 50, n), 1)
        }
        return pd.DataFrame(data)

    def entrenar_modelo(self, df):
        # Codificar variables categóricas
        for col in ["origen", "destino", "dia_semana", "congestion"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop("duracion_min", axis=1)
        y = df["duracion_min"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        self.modelo.fit(X_train, y_train)

        score = self.modelo.score(X_test, y_test)
        print(f"Precisión del modelo (R^2): {score:.2f}")

    def predecir_duracion(self, origen, destino, hora_dia, dia_semana, distancia_km, congestion):
        # Codificar datos de entrada
        entrada = {
            "origen": self.label_encoders["origen"].transform([origen])[0],
            "destino": self.label_encoders["destino"].transform([destino])[0],
            "hora_dia": hora_dia,
            "dia_semana": self.label_encoders["dia_semana"].transform([dia_semana])[0],
            "distancia_km": distancia_km,
            "congestion": self.label_encoders["congestion"].transform([congestion])[0]
        }

        df_entrada = pd.DataFrame([entrada])
        prediccion = self.modelo.predict(df_entrada)[0]
        return round(prediccion, 2)

# 🧪 Ejemplo de uso:
sistema = SistemaTransporteInteligente()

# 1. Generar y entrenar modelo con datos simulados
df = sistema.generar_dataset_simulado()
sistema.entrenar_modelo(df)

# 2. Hacer predicción con nuevos datos
origen = "A"
destino = "F"
hora_dia = 9
dia_semana = "Lunes"
distancia_km = 12.0
congestion = "Alta"

duracion_estimada = sistema.predecir_duracion(origen, destino, hora_dia, dia_semana, distancia_km, congestion)
print(f"Duración estimada entre {origen} y {destino}: {duracion_estimada} minutos")
