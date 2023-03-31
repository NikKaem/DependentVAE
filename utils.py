import torch; torch.manual_seed(0)
import torch.utils
import torch.distributions
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

def get_cov_dataset(
        dset,
        N,
        N_valid,
        incl_inds=False,
        lam=0.5,
        chol_sparsity=0.0,
        chol_min=0.5,
        chol_max=0.99,
        extra_seed=None,
        sparsify=0,
        bs=128,
        **kwargs,
):
    """extra seed not necessary if pl.seed_everything is set"""
    print("ignoring parameters: ", kwargs)
    dset_fn = dict(
        sine=to_sine,
        ccube=to_crescent_cube,
        cres=to_crescent,
        abs=to_abs,
        sign=to_sign,
    )[dset]

    cov = make_spd_matrix(
        n_dim=N,
        random_state=extra_seed,
    )

    # Doesn't seem to work as a spd matrix even with adding a small epsilon. Seems like a rounding issue but cannot pin it down
    """cov = make_sparse_spd_matrix(
        dim=N,
        alpha=chol_sparsity,
        smallest_coef=chol_min,
        largest_coef=chol_max,
        random_state=extra_seed,
        norm_diag=True,
    )"""

    if sparsify > 0:
        cov[np.abs(cov) < sparsify] = 0

    final_cov = lam * np.eye(N) + (1 - lam) * cov
    X = np.random.multivariate_normal(mean=np.zeros(N), cov=final_cov, size=2).T
    X = torch.from_numpy(X).float()
    xt = dset_fn(X)
    if incl_inds:
        xt = TensorDataset(
            xt,
            torch.arange(len(xt)),
        )
    tl = DataLoader(xt, batch_size=bs, shuffle=True)

    xv = dset_fn(torch.randn(N_valid, 2))
    vl = DataLoader(xv, batch_size=bs, shuffle=False)
    xtt = dset_fn(torch.randn(N_valid, 2))
    ttl = DataLoader(xtt, batch_size=bs, shuffle=False)

    return (
        tl,
        vl,
        ttl,
        dict(cov=cov, spectral=None, lam=lam, D=2, data_type="2d"),
    )


def to_sign(X):
    x1 = X[:, 0]
    x2_mean = torch.sign(x1) + x1
    x2_var = torch.exp(-3 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_abs(X):
    x1 = X[:, 0]
    x2_mean = torch.abs(x1) - 1
    x2_var = torch.exp(-3 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_crescent(X):
    x1 = X[:, 0]
    x2_mean = 0.5 * x1**2 - 1
    x2_var = torch.exp(-2 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_sine(X):
    x1 = X[:, 0]
    x2_mean = torch.sin(5 * x1)
    x2_var = torch.exp(-2 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_crescent_cube(X):
    x1 = X[:, 0]
    x2_mean = 0.2 * x1**3
    x2_var = torch.ones(x1.shape)
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x2, x1)).t()
    return x


def get_torch_generator():
    g = torch.Generator()
    global_seed = torch.initial_seed() % 2**32
    g.manual_seed(global_seed)
    return g


def plot_2d(
        samples_original,
        samples_dvae,
        samples_vae,
        titles,
        pixels=300,
        dpi=96,
        bounds=[[-3, 3], [-3, 3]],
        bins=256,
):
    if isinstance(samples_original, torch.Tensor):
        samples_original = samples_original.cpu().numpy()
    if isinstance(samples_dvae, torch.Tensor):
        samples_dvae = samples_dvae.cpu().numpy()
    if isinstance(samples_vae, torch.Tensor):
        samples_vae = samples_vae.cpu().numpy()

    figure, axes = plt.subplots(1,3, figsize=(pixels*3 / dpi, pixels / dpi), dpi=dpi, sharey=True)

    for i, (sample, title) in enumerate(zip([samples_original, samples_dvae, samples_vae], titles)):
        axes[i].hist2d(
            sample[..., 0], sample[..., 1], bins=bins, range=bounds
        )
        axes[i].set_title(title)

    plt.xlim(bounds[0])
    plt.ylim(bounds[1])

    plt.show()
    plt.close()