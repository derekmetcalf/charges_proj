import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor

"""
Partitioning the electron density of a molecule (or collection of molecules)
is of fundamental importance in simulating molecular motion using molecular
dynamics. Traditionally, partitioning the electron density requires a 
relatively expensive wavefunction or density functional theory (DFT) calculation,
methods that formally scale as at least O(N^3). By contrast, the underlying
molecular dynamics scales approximately as O(N), making the DFT the
rate-limiting step.

Here, we construct an empirical model of atomic charge partitioning with
asymptotic O(N^2) cost, reducing the formal scaling of the entire method
by O(N).

To accomplish this, we map an atom in its environment to a corresponding
scalar, its atomic charge. The atomic charge label is computed by the minimal
basis iterative stockholder (MBIS) algorithm performed on a second-order
perturbative wavefunction from the simplest truncation of symmetry-adapted
perturbation theory (SAPT0).

i.e. We seek the function that maps the atomic environment (X) to its
corresponding atomic charge (y):
        X <- y
Here, choosing X (Coulomb matrices, symmetry functions, graph convolutions),
and an appropriate functional form (KRR, k-NN, MLP) are critical in achieving
a sufficiently fast and physically meaningful mapping.

"""

def reconstitute(RAp, RBp, ZAp, ZBp, QAp, QBp):
    """
    Args: padded individual monomer coordinates, atom nums, and charges
    Returns: unpadded, concatenated coords, atom nums, and charges
    """
    R = []
    Z = []
    Q = []
    for i in range(len(RAp)):
        R_mol = []
        Z_mol = []
        Q_mol = []
        for j in range(len(RAp[i])):
            if ZAp[i][j] != 0:
                R_mol.append(RAp[i][j])
                Z_mol.append(ZAp[i][j])
                Q_mol.append(QAp[i][j])
        for j in range(len(RBp[i])):
            if ZBp[i][j] != 0:
                R_mol.append(RBp[i][j])
                Z_mol.append(ZBp[i][j])
                Q_mol.append(QBp[i][j])
        R.append(np.array(R_mol))
        Z.append(np.array(Z_mol))
        Q.append(np.array(Q_mol))
    return R, Z, Q

def get_local_coulomb(R, Z, n=5):
    """
    Generates local Coulomb matrix for set of molecules
    Args: Unpadded list of cartesian coords, atom nums
        optional: number of neighbors to consider default=5
    Returns: Unrolled list of local atomic coulomb matrices
    """
    C = []
    dists = []
    for i, coords in enumerate(R):
        dists = distance_matrix(coords, coords)
        for j in range(dists.shape[0]):
            inds = dists[j].argsort()[:n]
            R_local = np.zeros((n,n))
            ZZ_local = np.zeros((n,n))
            for k in range(len(inds)):
                for l in range(len(inds)):
                    if k != l:
                        R_local[k][l] = dists[inds[k],inds[l]]
                    elif k == l:
                        R_local[k][l] = 1.0 #fixes diagonal inf
                    ZZ_local[k][l] = Z[i][inds[k]] * Z[i][inds[l]]
            C.append(ZZ_local / R_local)
    C = np.array(C)
    return np.reshape(C, (C.shape[0], C.shape[1]**2))

def get_symmetry_functions(R, Z, eta=16., num = 20, r_c = 5.):
    """
    Generates weighted radial symmetry functions for each atom in a set of
    molecules a la Behler, Parrinello, Gastegger, Marquetand.
    Args: Unpadded list of Cartesian coords, atom nums
        optional:
         eta      : hyperparameter, corresponding to Gaussian widths
         mu_space : hyperparameter, corresponding to space between Gaussians
         r_c      : hyperparameter, corresponding to cutoff radius
    Returns: Unrolled list of atomic radial symmetry functions
    """
    Z_dict = {1.0 : 0,
              6.0 : 1,
              7.0 : 2,
              8.0 : 3,
              9.0 : 4,
              16.0: 5,}

    G = []
    mu = np.linspace(0.8, r_c, num)
    for i, coords in enumerate(R):
        dists = distance_matrix(coords, coords)
        for j in range(dists.shape[0]):
            atom_G = np.zeros(len(mu)*len(Z_dict))
            for k in range(dists.shape[0]):
                if j != k:
                    if dists[j][k] < r_c:
                        cutoff = 0.5 * (np.cos(dists[j][k] / r_c) + 1)
                    else: cutoff = 0.0
                    exp_term = np.exp(-eta * dists[j][k] - mu) ** 2
                    ind = Z_dict[Z[i][k]] * len(mu)
                    atom_G[ind:ind+len(mu)] += exp_term * cutoff
            G.append(atom_G)
    return np.array(G)

if __name__ == "__main__":
    # load in system names, electrons, atomic numbers, and coordinates
    # TODO: pandafy
    names = np.load("bms_data/names_train.npy")
    e_A = np.load("bms_data/Q_A_train.npy")
    e_B = np.load("bms_data/Q_B_train.npy")
    Z_A = np.load("bms_data/Z_A_train.npy")
    Z_B = np.load("bms_data/Z_B_train.npy")
    R_A = np.load("bms_data/R_A_train.npy")
    R_B = np.load("bms_data/R_B_train.npy")
    
    Q_A = Z_A - e_A
    Q_B = Z_B - e_B
    
    # these are split by monomer and padded, reconstituting total system w/o pad
    R, Z, Q = reconstitute(R_A, R_B, Z_A, Z_B, Q_A, Q_B)
    
    ## generate atomic environment descriptors
    #X = get_local_coulomb(R, Z, n=5)
    X = get_symmetry_functions(R, Z, eta=1., num=25)
    print(X[0])
    print(X.shape)
    print(X.max())
    #exit()
    #C = np.reshape(C_mat, (C_mat.shape[0], C_mat.shape[1]**2))

    ##The following feature represents the null hypothesis, where atoms
    ## are defined only by their atomic number. As you would expect,
    ## this doesn't work very well.
    #C = np.expand_dims(np.hstack(Z), -1)
    ##

    y = np.hstack(Q)

    mlp = MLPRegressor(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=10)
    scoring = ('r2', 'neg_mean_absolute_error')
    cv_results = cross_validate(mlp, X, y, cv=3, scoring=scoring, return_train_score=True)
    
    print(cv_results)
