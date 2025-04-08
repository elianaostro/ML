import numpy as np
from collections import Counter

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, lambda_=0.1, class_weight=None):
        """
        Inicializa el modelo de regresión logística con regularización L2.
        
        Parámetros:
        - learning_rate: Tasa de aprendizaje para el descenso de gradiente
        - n_iter: Número de iteraciones de entrenamiento
        - lambda_: Parámetro de regularización L2
        - class_weight: Diccionario con pesos para cada clase {0: w0, 1: w1}
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.class_weight = class_weight
        self.weights = None
        self.bias = None
        self.classes_ = None
    
    def _sigmoid(self, z):
        """Función sigmoide para mapear valores a probabilidades entre 0 y 1"""
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, h, y, weights):
        """Función de costo con regularización L2"""
        m = len(y)
        l2_reg = (self.lambda_ / (2 * m)) * np.sum(weights**2)
        
        if self.class_weight is not None:
            weight_vector = np.where(y == 1, self.class_weight[1], self.class_weight[0])
            return (-1/m) * np.sum(weight_vector * (y * np.log(h + 1e-15) + (1-y) * np.log(1-h + 1e-15))) + l2_reg
        else:
            return (-1/m) * np.sum(y * np.log(h + 1e-15) + (1-y) * np.log(1-h + 1e-15)) + l2_reg
    
    def fit(self, X, y):
        """Entrena el modelo con los datos X e y"""
        # Convertir y a numpy array si no lo es
        y = np.array(y)
        
        # Guardar clases únicas
        self.classes_ = np.unique(y)
        
        # Si class_weight es 'balanced', calcular pesos automáticamente
        if self.class_weight == 'balanced':
            n_samples = len(y)
            n_classes = len(self.classes_)
            class_count = {0: np.sum(y == 0), 1: np.sum(y == 1)}
            
            # Evitar división por cero
            if class_count[0] == 0 or class_count[1] == 0:
                self.class_weight = None
            else:
                self.class_weight = {
                    0: n_samples / (n_classes * class_count[0]),
                    1: n_samples / (n_classes * class_count[1])
                }
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(linear_model)
            
            # Calcular gradientes
            if self.class_weight is not None:
                weight_vector = np.where(y == 1, self.class_weight[1], self.class_weight[0])
                dw = (1/n_samples) * np.dot(X.T, weight_vector * (h - y)) + (self.lambda_ / n_samples) * self.weights
            else:
                dw = (1/n_samples) * np.dot(X.T, (h - y)) + (self.lambda_ / n_samples) * self.weights
            
            db = (1/n_samples) * np.sum(h - y)
            
            # Actualizar parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        """Devuelve las probabilidades predichas para cada clase"""
        linear_model = np.dot(X, self.weights) + self.bias
        proba_class1 = self._sigmoid(linear_model)
        return np.column_stack((1 - proba_class1, proba_class1))
    
    def predict(self, X, threshold=0.5):
        """Predice las clases para los datos X"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, lambda_=0.1):
        """
        Inicializa el modelo de regresión logística multinomial con regularización L2.
        
        Parámetros:
        - learning_rate: Tasa de aprendizaje para el descenso de gradiente
        - n_iter: Número de iteraciones de entrenamiento
        - lambda_: Parámetro de regularización L2
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.weights = None
        self.classes_ = None
    
    def _softmax(self, z):
        """Función softmax para clasificación multiclase"""
        # Estabilidad numérica: restar el máximo para evitar overflow
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _loss(self, h, y, weights):
        """Función de costo con regularización L2"""
        m = y.shape[0]
        l2_reg = (self.lambda_ / (2 * m)) * np.sum(weights**2)
        return (-1/m) * np.sum(y * np.log(h + 1e-15)) + l2_reg
    
    def fit(self, X, y):
        """Entrena el modelo con los datos X e y"""
        # Convertir y a one-hot encoding
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_onehot = np.zeros((len(y), n_classes))
        
        for i, cls in enumerate(self.classes_):
            y_onehot[y == cls, i] = 1
        
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, n_classes))
        
        # Descenso de gradiente
        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights)
            h = self._softmax(linear_model)
            
            # Calcular gradiente
            dw = (1/n_samples) * np.dot(X.T, (h - y_onehot)) + (self.lambda_ / n_samples) * self.weights
            
            # Actualizar pesos
            self.weights -= self.learning_rate * dw
    
    def predict_proba(self, X):
        """Devuelve las probabilidades predichas para cada clase"""
        linear_model = np.dot(X, self.weights)
        return self._softmax(linear_model)
    
    def predict(self, X):
        """Predice las clases para los datos X"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

class LDA:
    def __init__(self):
        """Inicializa el modelo de Análisis Discriminante Lineal"""
        self.means = None
        self.cov = None
        self.priors = None
        self.classes_ = None
    
    def fit(self, X, y):
        """Entrena el modelo con los datos X e y"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.means = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calcular medias y priors por clase
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means[i] = np.mean(X_c, axis=0)
            self.priors[i] = X_c.shape[0] / X.shape[0]
        
        # Calcular matriz de covarianza común
        self.cov = np.zeros((n_features, n_features))
        for c in range(n_classes):
            X_c = X[y == self.classes_[c]]
            diff = X_c - self.means[c]
            self.cov += diff.T @ diff
        self.cov /= (X.shape[0] - n_classes)
    
    def _multivariate_normal_pdf(self, X, mean, cov):
        """Calcula la PDF multivariada normal"""
        n_features = X.shape[1]
        diff = X - mean
        cov_inv = np.linalg.pinv(cov)
        
        # Calcular el exponente
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        
        # Calcular la constante de normalización
        norm_const = 1.0 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(np.linalg.det(cov)))
        
        return norm_const * np.exp(exponent)
    
    def predict_proba(self, X):
        """Devuelve las probabilidades predichas para cada clase"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            proba[:, i] = self._multivariate_normal_pdf(X, self.means[i], self.cov) * self.priors[i]
        
        # Normalizar las probabilidades
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        return proba
    
    def predict(self, X):
        """Predice las clases para los datos X"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        """
        Inicializa el árbol de decisión.
        
        Parámetros:
        - max_depth: Profundidad máxima del árbol
        - min_samples_split: Mínimo número de muestras para dividir un nodo
        - criterion: Criterio para medir la calidad de la división ('gini' o 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
    
    def _gini(self, y):
        """Calcula la impureza de Gini para un conjunto de etiquetas"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities**2)
    
    def _entropy(self, y):
        """Calcula la entropía para un conjunto de etiquetas"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """Calcula la ganancia de información para un split dado"""
        if self.criterion == 'gini':
            parent_impurity = self._gini(y)
        else:  # entropy
            parent_impurity = self._entropy(y)
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = n_left + n_right
        
        if n_left == 0 or n_right == 0:
            return 0
        
        if self.criterion == 'gini':
            left_impurity = self._gini(y[left_mask])
            right_impurity = self._gini(y[right_mask])
        else:  # entropy
            left_impurity = self._entropy(y[left_mask])
            right_impurity = self._entropy(y[right_mask])
        
        child_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """Encuentra el mejor split para un nodo"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Construye el árbol recursivamente"""
        n_samples = X.shape[0]
        
        # Criterios de parada
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return {'class': Counter(y).most_common(1)[0][0], 'is_leaf': True}
        
        # Encontrar el mejor split
        feature, threshold, best_gain = self._best_split(X, y)
        
        if feature is None or best_gain <= 0:  # No se pudo encontrar un split útil
            return {'class': Counter(y).most_common(1)[0][0], 'is_leaf': True}
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Construir subárboles
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'is_leaf': False
        }
    
    def fit(self, X, y):
        """Entrena el modelo con los datos X e y"""
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, sample, tree):
        """Predice una sola muestra recursivamente"""
        if tree['is_leaf']:
            return tree['class']
        
        if sample[tree['feature']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])
    
    def predict(self, X):
        """Predice las clases para los datos X"""
        return np.array([self._predict_sample(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 max_features=None, criterion='gini', random_state=None):
        """
        Inicializa el bosque aleatorio.
        
        Parámetros:
        - n_estimators: Número de árboles en el bosque
        - max_depth: Profundidad máxima de cada árbol
        - min_samples_split: Mínimo número de muestras para dividir un nodo
        - max_features: Número máximo de características a considerar para cada split
        - criterion: Criterio para medir la calidad de la división ('gini' o 'entropy')
        - random_state: Semilla para reproducibilidad
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
    
    def _bootstrap_sample(self, X, y):
        """Crea una muestra bootstrap"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Entrena el modelo con los datos X e y"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.trees = []
        n_features = X.shape[1]
        
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
        
        for _ in range(self.n_estimators):
            # Muestra bootstrap
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Selección aleatoria de características
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            self.feature_indices.append(feature_indices)
            
            # Entrenar árbol con las características seleccionadas
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split,
                              criterion=self.criterion)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        """Predice las clases mediante votación mayoritaria"""
        predictions = np.zeros((X.shape[0], len(self.trees)))
        
        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            predictions[:, i] = tree.predict(X[:, feature_idx])
        
        # Votación mayoritaria
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions.astype(int)])
    
    def predict_proba(self, X):
        """Devuelve las probabilidades predichas para cada clase"""
        predictions = np.zeros((X.shape[0], len(self.trees)))
        
        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            predictions[:, i] = tree.predict(X[:, feature_idx])
        
        # Calcular probabilidades como la frecuencia de cada clase en los árboles
        n_classes = len(np.unique(predictions))
        proba = np.zeros((X.shape[0], n_classes))
        
        for i in range(X.shape[0]):
            class_counts = Counter(predictions[i, :].astype(int))
            for cls, count in class_counts.items():
                proba[i, cls] = count / self.n_estimators
        
        return proba