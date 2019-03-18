import torch
import torch.nn as nn

def kl_div(p_1, p_2):
  return p_1 * torch.log(p_1 / p_2)

def calc_dist_cond_protected(data, target, protected_col_idx, protected_val):
  matches = data[:, protected_col_idx] == protected_val
  return target[matches].float().sum() / (matches).float().sum()

def calc_fairness(data, target, protected_col_idx=-1):
  protected = calc_dist_cond_protected(data, target, protected_col_idx, 1)
  unprotected = calc_dist_cond_protected(data, target, protected_col_idx, 0)
  return kl_div(protected, unprotected)

def calc_pred_fairness(data, preds, protected_col_idx=-1):
  protected_rows = data[:, protected_col_idx] == 1
  unprotected_rows = data[:, protected_col_idx] == 0
  protected = preds[protected_rows].float().sum() / (protected_rows).float().sum()
  unprotected = preds[unprotected_rows].float().sum() / (unprotected_rows).float().sum()
  return kl_div(protected, unprotected)
