import torch
import torch.nn as nn

def kl_div(p_1, p_2):
  t_1 = p_1 * torch.log(p_1 / p_2)
  t_2 = (1 - p_1) * torch.log((1 - p_1) / (1 - p_2))
  return t_1 + t_2

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

def assess_influence_retrain(trainer,
                             data,
                             target,
                             reg=0.0,
                             lim=None,
                             batch_size=1000,
                             num_epochs=10,
                             verbose=True):
  fairness = []
  for idx in range(min(len(data), lim) if lim is not None else len(data)):
    trainer.retrain_leave_one_out(data, target, idx, reg, batch_size, num_epochs, verbose)
    trainer.model.eval()
    preds = trainer.model(data) > 0.5
    fairness.append(calc_pred_fairness(data, preds).item())
  return torch.tensor(fairness)

def make_more_fair_retrain(trainer, data, target, num_to_drop=100, reg=0.1, batch_size=1000, num_epochs=1, verbose=False):
  influences = assess_influence_retrain(trainer, data, target, reg=reg, batch_size=batch_size, num_epochs=num_epochs, verbose=verbose)
  infs, idxs = torch.sort(influences)
  trainer.retrain_leave_one_out(data, target, idxs[:num_to_drop], reg=reg, batch_size=batch_size, num_epochs=num_epochs, verbose=verbose)
