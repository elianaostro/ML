import numpy as np

def get_data(N=10, noise=0.1, X_interval=(0,1), gt_fn=None, seed=42):
    generator = np.random.RandomState(seed=seed)
    if not gt_fn:
        raise ValueError("You must define a ground truth function")
    X = generator.uniform(*X_interval, size=N)
    Y_true = gt_fn(X)
    Y = Y_true + noise*generator.standard_normal(Y_true.shape)
    return X, Y, Y_true

def get_best_coef(X, Y, M, basis_fn=None):
    if basis_fn is None:
        # Default to polynomial basis if none provided
        basis_fn = lambda x, degree: x**degree

    # Construct the design matrix using the basis function
    PHI = np.zeros((len(X), M))
    for i in range(M):
        PHI[:, i] = basis_fn(X, i)
    
    # Solve for the coefficients using the normal equation
    PHI_T_PHI = np.dot(PHI.T, PHI)
    PHI_T_Y = np.dot(PHI.T, Y)
    PHI_T_PHI_inv = np.linalg.inv(PHI_T_PHI)  
    w = np.dot(PHI_T_PHI_inv, PHI_T_Y)

    print(f"Polinomio grado {M-1}: {np.polynomial.polynomial.Polynomial(w)}")
    print(f"ϕ: \n{PHI}")
    print(f"W:\n {w}")
    
    return w

def model_predict(w, X, basis_fn=None):
    if basis_fn is None:
        basis_fn = lambda x, degree: x**degree
    
    PHI = np.zeros((len(X), len(w)))
    for i in range(len(w)):
        PHI[:, i] = basis_fn(X, i)
    
    return np.dot(PHI, w)