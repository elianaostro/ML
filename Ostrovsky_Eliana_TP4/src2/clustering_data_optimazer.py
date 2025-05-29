
# Funciones auxiliares para métricas
def metrica_inercia(modelo, X):
    """Función de métrica para K-means basada en inercia."""
    if hasattr(modelo, 'inercia_'):
        return modelo.inercia_
    else:
        return modelo.calcular_inercia(X)

def metrica_log_likelihood(modelo, X):
    """Función de métrica para GMM basada en log-likelihood negativa."""
    if hasattr(modelo, 'log_likelihood_'):
        return -modelo.log_likelihood_  # Negativa porque queremos maximizar
    else:
        return -modelo.log_likelihood(X)


class ClusteringAnalyzer:
    """Clase para análisis y comparación de algoritmos de clustering."""
    
    def __init__(self, max_k=20, min_k=1):
        self.max_k = max_k
        self.min_k = min_k
        self.resultados_ = {}
        self.mejor_k_ = None
        self.mejor_modelo_ = None
    
    def metodo_codo(self, X, clustering_class=KMeans, k_range=None, **kwargs):
        """
        Implementa el método del codo para determinar el número óptimo de clusters.
        
        Args:
            X: Datos de entrada
            clustering_class: Clase del algoritmo (KMeans o GMM)
            k_range: Rango de k a probar (por defecto usa min_k a max_k)
            **kwargs: Argumentos adicionales para el constructor del algoritmo
        
        Returns:
            dict: Diccionario con k, inercias, mejor_k, mejor_modelo y figura
        """
        if k_range is None:
            k_range = range(self.min_k, self.max_k + 1)
        
        print(f"Ejecutando método del codo para {clustering_class.__name__}...")
        print(f"Probando k desde {min(k_range)} hasta {max(k_range)}")
        
        ks = list(k_range)
        inercias = []
        modelos = []
        
        # Calcular inercia para cada k
        for k in ks:
            print(f"\nProbando k={k}:")
            modelo = clustering_class(k=k, **kwargs)
            modelo.fit(X)
            
            # Calcular inercia (distancia total a centroides)
            if hasattr(modelo, 'inercia_'):
                inercia = modelo.inercia_
            else:
                # Para GMM, calculamos inercia basada en las medias como centroides
                if hasattr(modelo, 'medias_'):
                    labels = modelo.predict(X)
                    inercia = sum(np.linalg.norm(X[i] - modelo.medias_[labels[i]]) ** 2 
                                for i in range(len(X)))
                else:
                    inercia = modelo.calcular_inercia(X)
            
            inercias.append(inercia)
            modelos.append(modelo)
            print(f"Inercia para k={k}: {inercia:.4f}")
        
        # Encontrar el mejor k usando el método del codo
        mejor_k = self._encontrar_codo(ks, inercias)
        mejor_modelo = modelos[ks.index(mejor_k)]
        
        # Crear gráficos
        fig = self._crear_graficos_codo(ks, inercias, mejor_k, 
                                       titulo=f"Método del codo ({clustering_class.__name__})")
        
        # Guardar resultados
        resultado = {
            'ks': ks,
            'inercias': inercias,
            'mejor_k': mejor_k,
            'mejor_modelo': mejor_modelo,
            'figura': fig,
            'modelos': modelos
        }
        
        self.resultados_[clustering_class.__name__] = resultado
        self.mejor_k_ = mejor_k
        self.mejor_modelo_ = mejor_modelo
        
        print(f"\n¡Análisis completado!")
        print(f"Mejor k encontrado: {mejor_k}")
        print(f"Inercia del mejor modelo: {inercias[ks.index(mejor_k)]:.4f}")
        
        return resultado
    
    def _encontrar_codo(self, ks, inercias):
        """
        Encuentra el 'codo' en la curva de inercias usando el método de la segunda derivada.
        """
        if len(inercias) < 3:
            return ks[0]
        
        # Calcular las diferencias (primera derivada)
        diff1 = np.diff(inercias)
        
        # Calcular las diferencias de las diferencias (segunda derivada)
        diff2 = np.diff(diff1)
        
        # Encontrar el punto donde la segunda derivada es máxima
        # (mayor cambio en la pendiente = codo)
        codo_idx = np.argmax(diff2) + 2  # +2 porque perdemos 2 puntos en las diferencias
        
        # Asegurar que el índice esté en rango
        codo_idx = min(codo_idx, len(ks) - 1)
        
        return ks[codo_idx]
    
    def _crear_graficos_codo(self, ks, inercias, mejor_k, titulo="Método del codo"):
        """Crea los gráficos del método del codo (completo y zoom)."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(titulo, fontsize=14, fontweight='bold')
        
        # Gráfico completo
        ax1.plot(ks, inercias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=mejor_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mejor k = {mejor_k}')
        ax1.set_xlabel("Número de clusters (k)", fontsize=12)
        ax1.set_ylabel("Inercia", fontsize=12)
        ax1.set_title("Vista completa", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Determinar rango para el zoom (±5 alrededor del mejor k)
        zoom_start = max(min(ks), mejor_k - 5)
        zoom_end = min(max(ks), mejor_k + 5)
        
        # Filtrar datos para el zoom
        zoom_indices = [i for i, k in enumerate(ks) if zoom_start <= k <= zoom_end]
        zoom_ks = [ks[i] for i in zoom_indices]
        zoom_inercias = [inercias[i] for i in zoom_indices]
        
        # Gráfico zoom
        if len(zoom_ks) > 1:
            ax2.plot(zoom_ks, zoom_inercias, 'ro-', linewidth=2, markersize=8)
            ax2.axvline(x=mejor_k, color='red', linestyle='--', alpha=0.7,
                       label=f'Mejor k = {mejor_k}')
            ax2.set_xlabel("Número de clusters (k)", fontsize=12)
            ax2.set_ylabel("Inercia", fontsize=12)
            ax2.set_title(f"Zoom (k = {zoom_start} a {zoom_end})", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No hay suficientes\npuntos para zoom', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Zoom no disponible", fontsize=12)
        
        plt.tight_layout()
        return fig


# Clase auxiliar para optimización de múltiples corridas
class ClusteringOptimizer:
    """Clase para encontrar el mejor modelo ejecutando múltiples intentos."""
    
    @staticmethod
    def optimizar_modelo(clustering_class, X, metric_func, n_trials=10, **kwargs):
        """
        Encuentra el mejor modelo ejecutando múltiples intentos.
        
        Args:
            clustering_class: Clase del algoritmo de clustering
            X: Datos de entrada
            metric_func: Función que toma el modelo y devuelve una métrica (menor es mejor)
            n_trials: Número de intentos
            **kwargs: Argumentos para el constructor de la clase
        """
        mejor_score = np.inf
        mejor_modelo = None
        
        for i in range(n_trials):
            modelo = clustering_class(**kwargs)
            modelo.fit(X)
            score = metric_func(modelo, X)
            
            if score < mejor_score:
                mejor_score = score
                mejor_modelo = modelo
            
            print(f"Intento {i+1}/{n_trials} - Score: {score:.4f}")
        
        print(f"Mejor score: {mejor_score:.4f}")
        return mejor_modelo


# Ejemplo de uso
def ejemplo_uso():
    """Ejemplo completo de cómo usar las clases de clustering."""
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    # Generar datos de ejemplo
    print("Generando datos de ejemplo...")
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    
    # Crear analizador
    analyzer = ClusteringAnalyzer(max_k=15, min_k=2)
    
    # Método 1: Analizar un solo algoritmo
    print("\n=== ANÁLISIS INDIVIDUAL ===")
    resultado_kmeans = analyzer.metodo_codo(X, KMeans, n_init=5)
    plt.show()  # Mostrar gráfico del método del codo
    
    # Método 2: Comparar múltiples algoritmos
    print("\n=== COMPARACIÓN DE ALGORITMOS ===")
    comparacion = analyzer.comparar_algoritmos(
        X, 
        algoritmos=[KMeans, GMM],
        k_range=range(2, 12),
        n_init=3  # Para K-means
    )
    plt.show()  # Mostrar gráfico comparativo
    
    # Graficar el mejor resultado
    print("\n=== VISUALIZACIÓN DEL MEJOR MODELO ===")
    fig_clusters = analyzer.graficar_clusters(X)
    plt.show()
    
    # Acceder al mejor modelo
    mejor_modelo = analyzer.mejor_modelo_
    print(f"\nEl mejor modelo es: {mejor_modelo.__class__.__name__} con k={mejor_modelo.k}")
    
    return analyzer, comparacion

# Función auxiliar para análisis rápido
def analisis_rapido(X, max_k=15):
    """Función para hacer un análisis rápido con configuración por defecto."""
    analyzer = ClusteringAnalyzer(max_k=max_k)
    comparacion = analyzer.comparar_algoritmos(X)
    
    # Mostrar todos los gráficos
    for resultado in comparacion['resultados'].values():
        resultado['figura'].show()
    
    comparacion['figura_comparacion'].show()
    analyzer.graficar_clusters(X).show()
    
    return analyzer.mejor_modelo_
    
def comparar_algoritmos(self, X, algoritmos=None, k_range=None, **kwargs):
    """
    Compara múltiples algoritmos de clustering usando el método del codo.
    
    Args:
        X: Datos de entrada
        algoritmos: Lista de clases de algoritmos (por defecto [KMeans, GMM])
        k_range: Rango de k a probar
        **kwargs: Argumentos adicionales para los constructores
    
    Returns:
        dict: Resultados de comparación para todos los algoritmos
    """
    if algoritmos is None:
        algoritmos = [KMeans, GMM]
    
    if k_range is None:
        k_range = range(self.min_k, self.max_k + 1)
    
    resultados_comparacion = {}
    
    print("=== COMPARACIÓN DE ALGORITMOS DE CLUSTERING ===\n")
    
    for algoritmo in algoritmos:
        print(f"\n{'='*50}")
        print(f"Analizando {algoritmo.__name__}")
        print(f"{'='*50}")
        
        resultado = self.metodo_codo(X, algoritmo, k_range, **kwargs)
        resultados_comparacion[algoritmo.__name__] = resultado
    
    # Crear gráfico comparativo
    fig_comparacion = self._crear_grafico_comparativo(resultados_comparacion)
    
    # Encontrar el mejor algoritmo y k global
    mejor_global = self._encontrar_mejor_global(resultados_comparacion)
    
    print(f"\n{'='*60}")
    print("RESUMEN DE COMPARACIÓN")
    print(f"{'='*60}")
    
    for nombre, resultado in resultados_comparacion.items():
        mejor_k = resultado['mejor_k']
        mejor_inercia = resultado['inercias'][resultado['ks'].index(mejor_k)]
        print(f"{nombre:15} | Mejor k: {mejor_k:2d} | Inercia: {mejor_inercia:8.2f}")
    
    print(f"\n🏆 GANADOR GLOBAL: {mejor_global['algoritmo']} con k={mejor_global['k']} (Inercia: {mejor_global['inercia']:.2f})")
    
    return {
        'resultados': resultados_comparacion,
        'figura_comparacion': fig_comparacion,
        'mejor_global': mejor_global
    }

def _crear_grafico_comparativo(self, resultados):
    """Crea un gráfico comparativo de todos los algoritmos."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colores = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (nombre, resultado) in enumerate(resultados.items()):
        color = colores[i % len(colores)]
        ks = resultado['ks']
        inercias = resultado['inercias']
        mejor_k = resultado['mejor_k']
        
        # Línea principal
        ax.plot(ks, inercias, 'o-', color=color, linewidth=2, 
                markersize=6, label=f'{nombre}', alpha=0.8)
        
        # Marcar el mejor k
        mejor_idx = ks.index(mejor_k)
        ax.scatter(mejor_k, inercias[mejor_idx], color=color, s=150, 
                    marker='*', edgecolors='black', linewidth=2,
                    label=f'{nombre} (k={mejor_k})')
    
    ax.set_xlabel("Número de clusters (k)", fontsize=12)
    ax.set_ylabel("Inercia", fontsize=12)
    ax.set_title("Comparación de Algoritmos - Método del Codo", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def _encontrar_mejor_global(self, resultados):
    """Encuentra el mejor algoritmo y k basado en la menor inercia."""
    mejor_inercia = float('inf')
    mejor_info = None
    
    for nombre, resultado in resultados.items():
        mejor_k = resultado['mejor_k']
        inercia = resultado['inercias'][resultado['ks'].index(mejor_k)]
        
        if inercia < mejor_inercia:
            mejor_inercia = inercia
            mejor_info = {
                'algoritmo': nombre,
                'k': mejor_k,
                'inercia': inercia,
                'modelo': resultado['mejor_modelo']
            }
    
    return mejor_info

def graficar_clusters(self, X, modelo=None, titulo=None):
    """
    Grafica los resultados del clustering.
    
    Args:
        X: Datos originales
        modelo: Modelo entrenado (usa el mejor modelo si no se especifica)
        titulo: Título del gráfico
    """
    import matplotlib.pyplot as plt
    
    if modelo is None:
        modelo = self.mejor_modelo_
        if modelo is None:
            raise ValueError("No hay modelo disponible. Ejecuta primero el análisis.")
    
    if titulo is None:
        titulo = f"Clustering - {modelo.__class__.__name__} (k={modelo.k})"
    
    # Obtener labels
    if hasattr(modelo, 'labels_'):
        labels = modelo.labels_
    else:
        labels = modelo.predict(X)
    
    # Obtener centroides si existen
    centroides = None
    if hasattr(modelo, 'centroides_'):
        centroides = modelo.centroides_
    elif hasattr(modelo, 'medias_'):
        centroides = modelo.medias_
    
    # Crear gráfico
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Colores para clusters
    labels_unicos = np.unique(labels)
    colores = plt.cm.tab10(np.linspace(0, 1, len(labels_unicos)))
    
    for i, label in enumerate(labels_unicos):
        puntos = X[labels == label]
        if label == -1:  # Ruido en DBSCAN
            ax.scatter(puntos[:, 0], puntos[:, 1], c='black', 
                        label="Ruido", s=30, alpha=0.6)
        else:
            ax.scatter(puntos[:, 0], puntos[:, 1], color=colores[i], 
                        label=f"Cluster {label}", s=50, alpha=0.7)
    
    # Graficar centroides si existen
    if centroides is not None:
        ax.scatter(centroides[:, 0], centroides[:, 1], color='black', 
                    marker='x', s=200, linewidths=3, label='Centroides')
    
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig