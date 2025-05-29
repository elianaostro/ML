
class GMM(ClusteringBase):
    """Implementación del algoritmo Gaussian Mixture Model (GMM)."""
    
    def __init__(self, k, max_iter=100, tol=1e-4, seed=0, reg_covar=1e-3):
        super().__init__(k, max_iter, tol, seed)
        self.reg_covar = reg_covar
        self.medias_ = None
        self.covs_ = None
        self.pesos_ = None
        self.log_likelihood_ = None
    
    def inicializar_gmm(self, X):
        """Inicializa parámetros del GMM usando K-means."""
        # Usar K-means para inicialización
        kmeans = KMeans(k=self.k, seed=self.seed, n_init=1)
        kmeans.fit(X)
        medias = kmeans.centroides_
        labels = kmeans.labels_
        
        n, d = X.shape
        pesos = np.bincount(labels, minlength=self.k) / n
        covs = []
        
        for j in range(self.k):
            Xj = X[labels == j]
            if Xj.shape[0] > 1:
                cov = np.cov(Xj, rowvar=False) + self.reg_covar * np.eye(d)
            else:
                cov = np.diag(np.var(X, axis=0)) * 0.1 + self.reg_covar * np.eye(d)
            covs.append(cov)
        
        return medias, covs, pesos
    
    def pdfs_gaussianas(self, X, medias=None, covs=None):
        """Calcula densidades de probabilidad gaussianas para todos los puntos."""
        medias = medias if medias is not None else self.medias_
        covs = covs if covs is not None else self.covs_
        
        n, d = X.shape
        k = medias.shape[0]
        pdfs = np.zeros((n, k))
        
        for j in range(k):
            diff = X - medias[j]
            inv = np.linalg.inv(covs[j])
            det = np.linalg.det(covs[j])
            norm = 1 / np.sqrt((2 * np.pi) ** d * det)
            exp = np.einsum('ni,ij,nj->n', diff, inv, diff)
            pdfs[:, j] = norm * np.exp(-0.5 * exp)
        
        return pdfs
    
    def expectation(self, X):
        """Paso E del algoritmo EM."""
        pdf_vals = self.pdfs_gaussianas(X)
        gamma = pdf_vals * self.pesos_
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
        return gamma
    
    def maximization(self, X, gamma):
        """Paso M del algoritmo EM."""
        n, d = X.shape
        Nk = gamma.sum(axis=0)
        
        # Actualizar pesos
        self.pesos_ = Nk / n
        
        # Actualizar medias
        self.medias_ = np.array([np.sum(gamma[:, j][:, None] * X, axis=0) / Nk[j] 
                                for j in range(self.k)])
        
        # Actualizar covarianzas
        covs = []
        for j in range(self.k):
            diff = X - self.medias_[j]
            cov = (gamma[:, j][:, None] * diff).T @ diff / Nk[j]
            cov += self.reg_covar * np.eye(d)
            covs.append(cov)
        self.covs_ = covs
    
    def log_likelihood(self, X):
        """Calcula la log-verosimilitud."""
        pdf_vals = self.pdfs_gaussianas(X)
        weighted_sum = np.dot(pdf_vals, self.pesos_)
        return np.sum(np.log(weighted_sum + 1e-10))
    
    def fit(self, X):
        """Entrena el modelo GMM usando el algoritmo EM."""
        print(f"Inicializando GMM con {self.k} clusters...")
        
        # Inicialización
        self.medias_, self.covs_, self.pesos_ = self.inicializar_gmm(X)
        n, d = X.shape
        prev_ll = -np.inf
        
        for i in range(self.max_iter):
            # Paso E
            gamma = self.expectation(X)
            
            # Paso M
            self.maximization(X, gamma)
            
            # Calcular log-likelihood
            ll = self.log_likelihood(X)
            self.update_progress_bar(i + 1, self.max_iter, metrics={'loglik': ll})
            
            # Verificar convergencia
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        
        # Asignar labels finales
        gamma = self.expectation(X)
        self.labels_ = np.argmax(gamma, axis=1)
        self.log_likelihood_ = ll
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predice clusters para nuevos puntos."""
        if not self.is_fitted_:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones.")
        
        gamma = self.expectation(X)
        return np.argmax(gamma, axis=1)
    
    def predict_proba(self, X):
        """Predice probabilidades de pertenencia a cada cluster."""
        if not self.is_fitted_:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones.")
        
        return self.expectation(X)

