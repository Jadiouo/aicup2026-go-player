"""Memory-lean generalised-LDA fit (same maths as player_sim3.fit_metric; chunked Sw).  X float32 (n,d) already standardised."""
import numpy as np, scipy.linalg
def fit_metric_lean(X, pid, ridge=3e-2, mmax=150, w_kappa=None):
    codes, inv = np.unique(pid, return_inverse=True); C, d = len(codes), X.shape[1]
    cnt = np.bincount(inv).astype(np.float64)
    mu = np.zeros((C, d))
    for a in range(0, len(X), 20000):
        sl = slice(a, a + 20000); np.add.at(mu, inv[sl], X[sl].astype(np.float64))
    mu /= cnt[:, None]
    Sw = np.zeros((d, d))
    for a in range(0, len(X), 20000):
        R = X[a:a + 20000].astype(np.float64) - mu[inv[a:a + 20000]]; Sw += R.T @ R
    Sw /= (len(X) - C); Sw += ridge * np.trace(Sw) / d * np.eye(d)
    gm = mu.mean(0); Sb = np.cov(mu.T)
    ev, V = scipy.linalg.eigh(Sb, Sw); o = np.argsort(-ev)
    return gm, V[:, o][:, :mmax], ev[o]
def project(Z, mu, sd, gm, V, chunk=20000):
    Y = np.empty((len(Z), V.shape[1]), np.float32)
    for a in range(0, len(Z), chunk): Y[a:a + chunk] = ((Z[a:a + chunk].astype(np.float64) - mu) / sd - gm) @ V
    return Y
