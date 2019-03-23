import torch
from sklearn.decomposition import PCA
import numpy as np
import scipy.linalg

from .influence import calc_log_reg_hessian_inverse, calc_log_reg_grad


def scatter_dists(group_1, group_2):
  data = np.concatenate([group_1, group_2])
  model = PCA(n_components=2)
  model.fit(data)
  return model.transform(group_1), model.transform(group_2)

def get_fisher_vectors(lr_model, data, target) -> np.ndarray:
  grads = calc_log_reg_grad(lr_model, data, target)
  hess_inv = calc_log_reg_hessian_inverse(lr_model, data)
  hess_inv_sqrt = torch.from_numpy(scipy.linalg.sqrtm(hess_inv).real).type_as(data)
  return grads.matmul(hess_inv_sqrt)
