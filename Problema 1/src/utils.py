def PCA(df, n_components=2):
    """
    PCA modificado para trabajar con DataFrames
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        n_components: Número de componentes a retener
        
    Returns:
        X_reduced (pd.DataFrame): Datos proyectados
        components (pd.DataFrame): Componentes principales
    """
    X = df.values
    cov = np.cov(X.T) 
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    components = eigenvectors[:, :n_components]
    X_reduced = np.dot(X, components)
    
    # Convertir a DataFrames con nombres descriptivos
    components_df = pd.DataFrame(
        components.T,
        columns=df.columns,
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    
    X_reduced_df = pd.DataFrame(
        X_reduced,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df.index
    )
    
    return X_reduced_df, components_df



print("\nVisualización de fronteras de decisión en espacio 2D")
X_train_2d, components_2d = PCA(X_train, n_components=2)

# LDA en 2D
lda_2d = LDA()
lda_2d.fit(X_train_2d, y_train)
plot_decision_boundary(lda_2d, X_val_2d, y_val, title="Frontera de Decisión - LDA")

# Regresión Logística en 2D
mlr_2d = LogisticRegression(learning_rate=0.1, n_iter=1000)
mlr_2d.fit(X_train_2d, y_train)
plot_decision_boundary(mlr_2d, X_val_2d, y_val, title="Frontera de Decisión - Regresión Logística")

importances = final_rf_model.feature_importances_
top_two_idx = np.argsort(importances)[-2:]
X_train_2d = X_train_array[:, top_two_idx]
X_val_2d = X_val_array[:, top_two_idx]


# Random Forest en 2D
rf_2d = RandomForest(n_estimators=50, max_depth=5, criterion='entropy')
rf_2d.fit(X_train_2d, y_train)
plot_decision_boundary(rf_2d, X_val_2d, y_val, title="Frontera de Decisión - Random Forest")

def KMeans(df, n_clusters=3, max_iter=100, random_state=None):
    """
    K-Means modificado para trabajar con DataFrames de pandas
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        n_clusters: Número de clusters
        max_iter: Máximo número de iteraciones
        random_state: Semilla aleatoria
        
    Returns:
        labels: Etiquetas de cluster para cada punto
        centroids: Posición de los centroides finales (DataFrame)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = df.values  # Convertir a array internamente
    n_samples = X.shape[0]
    
    # Inicialización de centroides usando índices del DataFrame
    random_indices = np.random.choice(df.index, n_clusters, replace=False)
    centroids = X[random_indices]
    
    for _ in range(max_iter):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        new_centroids = np.array([
            X[labels == k].mean(axis=0) for k in range(n_clusters)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Convertir centroides a DataFrame con nombres originales
    centroids_df = pd.DataFrame(centroids, columns=df.columns)
    
    return labels, centroids_df


def detect_outliers_iqr(X, factor=1.5):
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).
    
    Parámetros:
    - X: Datos de características (numpy array)
    - factor: Factor multiplicador del IQR (típicamente 1.5)
    
    Retorna:
    - Máscara booleana de outliers (True = outlier)
    """
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return (X < lower_bound) | (X > upper_bound)