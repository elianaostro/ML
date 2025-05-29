
class KMeans(ClusteringBase):
    """Implementación del algoritmo K-Means."""
    
    def __init__(self, k, max_iter=100, tol=1e-4, seed=0, n_init=1):
        super().__init__(k, max_iter, tol, seed)
        self.n_init = n_init
        self.inercia_ = None
    
    def actualizar_centroides(self, X, labels, k=None):
        """Actualiza los centroides calculando la media de cada cluster."""
        k = k or self.k
        return np.array([X[labels == i].mean(axis=0) for i in range(k)])
    
    def _kmeans_single(self, X, seed=None):
        """Ejecuta una sola corrida de K-means."""
        centroides = self.inicializar_centroides(X, seed=seed)
        
        for i in range(self.max_iter):
            labels = self.asignar_por_distancia(X, centroides)
            nuevos_centroides = self.actualizar_centroides(X, labels)
            
            if np.allclose(centroides, nuevos_centroides, atol=self.tol):
                break
                
            centroides = nuevos_centroides
        
        inercia = self.calcular_inercia(X, centroides, labels)
        return centroides, labels, inercia
    
    def fit(self, X):
        """Entrena el modelo K-means con múltiples inicializaciones."""
        best_inercia = np.inf
        best_result = None
        
        print(f"Entrenando K-Means con k={self.k}...")
        
        for i in range(self.n_init):
            seed = None if self.seed is None else self.seed + i
            resultado = self._kmeans_single(X, seed=seed)
            
            if resultado[2] < best_inercia:
                best_inercia = resultado[2]
                best_result = resultado
            
            self.update_progress_bar(i + 1, self.n_init, metrics={'inercia': resultado[2]})
        
        self.centroides_, self.labels_, self.inercia_ = best_result
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Asigna puntos a clusters basándose en los centroides entrenados."""
        if not self.is_fitted_:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones.")
        return self.asignar_por_distancia(X)

