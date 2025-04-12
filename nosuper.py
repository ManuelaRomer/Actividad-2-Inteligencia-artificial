import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

class SistemaTransporteNoSupervisado:
    def __init__(self):
        self.kmeans = None
        self.label_encoders = {}
        self.scaler = None

    def generar_dataset_simulado(self, n=200):
        data = {
            "origen": np.random.choice(["A", "B", "C", "D", "E"], n),
            "destino": np.random.choice(["B", "C", "D", "E", "F"], n),
            "hora_dia": np.random.randint(5, 22, n),
            "dia_semana": np.random.choice(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"], n),
            "distancia_km": np.round(np.random.uniform(2, 15, n), 2),
            "congestion": np.random.choice(["Baja", "Media", "Alta"], n)
        }
        return pd.DataFrame(data)

    def entrenar_modelo(self, df, n_clusters=3):
        # Codificar variables categóricas
        for col in ["origen", "destino", "dia_semana", "congestion"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Escalar los datos
        self.scaler = StandardScaler()
        datos_escalados = self.scaler.fit_transform(df)

        # Entrenar el modelo K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(datos_escalados)

        # Asignar clúster a cada fila
        df["grupo"] = self.kmeans.labels_

        print(f"Modelo entrenado con {n_clusters} grupos.")
        print(df.groupby("grupo").mean(numeric_only=True))

        # Graficar dos variables por simplicidad
        plt.scatter(df["distancia_km"], df["hora_dia"], c=df["grupo"], cmap='viridis')
        plt.xlabel("Distancia (km)")
        plt.ylabel("Hora del día")
        plt.title("Agrupamiento de trayectos")
        plt.show()

    def predecir_grupo(self, origen, destino, hora_dia, dia_semana, distancia_km, congestion):
        # Codificar entrada
        entrada = {
            "origen": self.label_encoders["origen"].transform([origen])[0],
            "destino": self.label_encoders["destino"].transform([destino])[0],
            "hora_dia": hora_dia,
            "dia_semana": self.label_encoders["dia_semana"].transform([dia_semana])[0],
            "distancia_km": distancia_km,
            "congestion": self.label_encoders["congestion"].transform([congestion])[0]
        }

        df_entrada = pd.DataFrame([entrada])
        entrada_escalada = self.scaler.transform(df_entrada)
        grupo = self.kmeans.predict(entrada_escalada)[0]
        return grupo
    
sistema = SistemaTransporteNoSupervisado()
df = sistema.generar_dataset_simulado()
sistema.entrenar_modelo(df, n_clusters=4)

# Probar con un trayecto nuevo
grupo = sistema.predecir_grupo("A", "F", 8, "Lunes", 10.0, "Alta")
print(f"Este trayecto pertenece al grupo/clúster: {grupo}")
