
class DBSCAN:
    """Implementación del algoritmo DBSCAN."""
    
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.is_fitted_ = False
    
    def vecinos(self, X, i):
        """Encuentra los vecinos de un punto dentro del radio eps."""
        dists = np.linalg.norm(X - X[i], axis=1)
        return np.where(dists <= self.eps)[0]
    
    def expandir_cluster(self, X, etiquetas, i, cluster_id):
        """Expande un cluster desde un punto core."""
        vecinos_i = self.vecinos(X, i)
        
        if len(vecinos_i) < self.min_samples:
            etiquetas[i] = -1  # Ruido
            return False
        
        # Marcar punto como parte del cluster
        etiquetas[i] = cluster_id
        vecinos_lista = list(vecinos_i)
        j = 0
        
        while j < len(vecinos_lista):
            p = vecinos_lista[j]
            
            if etiquetas[p] == -1:  # Era ruido, ahora es borde
                etiquetas[p] = cluster_id
            elif etiquetas[p] == 0:  # No visitado
                etiquetas[p] = cluster_id
                nuevos_vecinos = self.vecinos(X, p)
                
                # Si es punto core, agregar sus vecinos
                if len(nuevos_vecinos) >= self.min_samples:
                    vecinos_lista.extend(nuevos_vecinos)
            
            j += 1
        
        return True
    
    def fit(self, X):
        """Entrena el modelo DBSCAN."""
        n = len(X)
        etiquetas = np.zeros(n, dtype=int)  # 0: no visitado, -1: ruido
        cluster_id = 0
        
        print(f"Ejecutando DBSCAN con eps={self.eps}, min_samples={self.min_samples}...")
        
        for i in range(n):
            if etiquetas[i] != 0:  # Ya visitado
                continue
            
            if self.expandir_cluster(X, etiquetas, i, cluster_id + 1):
                cluster_id += 1
        
        self.labels_ = etiquetas
        self.is_fitted_ = True
        print(f"DBSCAN completado. Clusters encontrados: {cluster_id}")
        return self
    
    def fit_predict(self, X):
        """Entrena el modelo y devuelve las etiquetas."""
        self.fit(X)
        return self.labels_

