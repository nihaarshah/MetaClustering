import numpy as np
import matplotlib.pyplot as plt

# Other sythetic generation:
# https://github.com/lovit/synthetic_dataset/blob/2518aeea2db03852a997e1ab807c59218ebd634d/soydata/data/base.py#L112


def sample_mu(rng, k, d):
    """Sample means of clusters"""
    return np.random.uniform(-rng, rng, (k, d))


def sample_cov(k, d, alpha, beta):
    """Sample covariance of clusters with method in 
    Yibo et. al"""
    cov = []
    for _ in range(k):
        # Initial c_j
        C = np.random.normal(0, 1, (d, d))
        # Magnitude of entries in cov matrix
        s = np.random.uniform(alpha, alpha+beta)
        # Orthoganilze C
        C = np.linalg.qr(C)[0]/s
        cov.append(C.T @ C)

    return cov


def sample_cov_eig(k, d, alpha, beta):
    """Sample covariance of clusters with
    eigenvalue method"""
    cov = []
    for _ in range(k):
        # Initialize c_j
        C = np.random.normal(0, 1, (d, d))
        # Orthoganalize C
        C = np.linalg.qr(C)[0]
        # Generate eigen values diagonal matrix
        eig = np.diag(np.random.uniform(alpha, alpha+beta, d))
        # Create cov matrix
        cov.append(C.T @ eig @ C)

    return cov


def generate_dataset(n, d, k, rng, alpha, beta):
    """Generate a dataset"""
    mu = sample_mu(rng, k, d)
    cov = sample_cov(k, d, alpha, beta)

    data = []
    labels = []
    for i in range(k):
        data.append(np.random.multivariate_normal(
            mean=mu[i, :], cov=cov[i], size=n))
        labels.append(np.repeat(i, n))

    return np.vstack(data), np.hstack(labels)


def nonlinear(x, p, q, alpha, beta):
    """Non-linear augmentation in Yibo et al."""
    kr = np.random.uniform(alpha, alpha+beta)
    # p, q = np.random.choice(range(len(x)), size=2)
    z = x[[p, q]]
    r = np.linalg.norm(z)
    t = np.pi*r/kr

    xp = np.cos(t)*x[p] + np.sin(t)*x[q]
    xq = -np.sin(t)*x[p] + np.cos(t)*x[q]
    x[p] = xp
    x[q] = xq

    return x


def generate_nonlinear_dataset(n, d, k, rng, alpha, beta, a_nl, b_nl):
    """Generate a non-linear dataset"""
    mu = sample_mu(rng, k, d)
    cov = sample_cov(k, d, alpha, beta)
    print(cov)
    data = []
    labels = []
    for i in range(k):
        data_ = np.random.multivariate_normal(
            mean=mu[i, :], cov=cov[i], size=n)
        for _ in range(4):
            p, q = np.random.choice(range(data_.shape[1]), size=2)
            for j in range(len(data_)):
                data_[j, :] = nonlinear(data_[j, :], p, q, a_nl, b_nl)
        data.append(data_)
        labels.append(np.repeat(i, n))

    return np.vstack(data), np.hstack(labels)


if __name__ == "__main__":
    data, labels = generate_dataset(n=25, d=2, k=4, rng=10, alpha=1, beta=2)

    plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)
    plt.show()
    data, labels = generate_nonlinear_dataset(n=25, d=2, k=4, rng=10,
                                              alpha=10, beta=2,
                                              a_nl=12, b_nl=2
                                              )

    plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)
    plt.show()
