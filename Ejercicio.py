#ACTIVIDAD 2 - INTELIGENCIA ARTIFICIAL
class SistemaExpertoRutas:
    def __init__(self):
        # Creamos la base de conocimiento la cual tiene las conexiones entre estaciones con sus tiempos
        self.base_conocimiento = {
            "A": {"B": 10, "D": 20},
            "B": {"C": 15, "E": 12},
            "C": {"F": 8},
            "D": {"E": 5},
            "E": {"C": 10, "F": 6},
        }
    
    def mejor_ruta(self, origen, destino, ruta_actual=[], tiempo_actual=0):
        """Encuentra la mejor ruta usando búsqueda basada en conocimiento"""
        if origen not in self.base_conocimiento:
            return None, float('inf')  

        ruta_actual = ruta_actual + [origen]

        if origen == destino:
            return ruta_actual, tiempo_actual  
        
        mejor_camino = None
        menor_tiempo = float('inf')

        for siguiente, tiempo in self.base_conocimiento[origen].items():
            if siguiente not in ruta_actual:  
                nueva_ruta, nuevo_tiempo = self.mejor_ruta(siguiente, destino, ruta_actual, tiempo_actual + tiempo)
                if nuevo_tiempo < menor_tiempo:
                    mejor_camino, menor_tiempo = nueva_ruta, nuevo_tiempo

        return mejor_camino, menor_tiempo

# ---- EJECUCIÓN DEL SISTEMA EXPERTO ----
sistema = SistemaExpertoRutas()

origen, destino = "A", "B"
ruta, tiempo = sistema.mejor_ruta(origen, destino)

if ruta:
    print(f"La mejor ruta de {origen} a {destino} es: {' → '.join(ruta)} con un tiempo de {tiempo} minutos.")
else:
    print(f"No hay una ruta disponible entre {origen} y {destino}.")

