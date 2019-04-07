import torch
import torch.nn as nn

from .influence import calc_s_tests, calc_influences

def kl_div(p_1, p_2):
  t_1 = p_1 * torch.log(p_1 / p_2)
  t_2 = (1 - p_1) * torch.log((1 - p_1) / (1 - p_2))
  return t_1 + t_2

def calc_dist_cond_protected(data, target, protected_col_idx, protected_val):
  matches = data[:, protected_col_idx] == protected_val
  return target[matches].float().sum() / (matches).float().sum()

def calc_fairness(data, target, protected_col_idx=-1):
  protected = calc_dist_cond_protected(data, target, protected_col_idx, 0)
  unprotected = calc_dist_cond_protected(data, target, protected_col_idx, 1)
  return kl_div(protected, unprotected)

def calc_pred_fairness(data, preds, protected_col_idx=-1):
  protected_rows = data[:, protected_col_idx] == 0
  unprotected_rows = data[:, protected_col_idx] == 1
  protected = preds[protected_rows].float().sum() / (protected_rows).float().sum()
  unprotected = preds[unprotected_rows].float().sum() / (unprotected_rows).float().sum()
  return kl_div(protected, unprotected)

def eval_fairness(model, data):
  model.eval()
  with torch.no_grad():
    # preds = (model(data) > 0.5)
    preds = model(data)
  return calc_pred_fairness(data, preds)

def assess_impact_retrain(trainer,
                          data,
                          target,
                          test_data,
                          reg=0.0,
                          lim=None,
                          batch_size=1000,
                          num_epochs=30,
                          verbose=True):
  fairness = []
  for idx in range(min(len(data), lim) if lim is not None else len(data)):
    trainer.retrain_leave_one_out(data, target, idx, reg, batch_size, num_epochs, verbose)
    fairness.append(eval_fairness(trainer.model, test_data).item())
  return torch.tensor(fairness)

def make_more_fair_retrain(trainer, data, target, test_data, num_to_drop=100, reg=0.1, batch_size=1000, num_epochs=30, verbose=False):
  impacts = assess_impact_retrain(trainer, data, target, test_data, reg=reg, batch_size=batch_size, num_epochs=num_epochs, verbose=verbose)
  infs, idxs = torch.sort(impacts)
  trainer.retrain_leave_one_out(data, target, idxs[:num_to_drop], reg=reg, batch_size=batch_size, num_epochs=num_epochs, verbose=verbose)
  return impacts

def assess_impact_influence(trainer,
                            data,
                            test_data,
                            test_target):
  s_tests = calc_s_tests(trainer.model, test_data, test_target)
  return calc_influences(trainer.model, s_tests, data)

def make_more_fair_influence(trainer, data, target, test_data, num_to_drop=100, reg=0.1, batch_size=1000, num_epochs=30, verbose=False):
  influences = assess_impact_influence(trainer, data, test_data)
  infs, idxs = torch.sort(influences)
  trainer.retrain_leave_one_out(data, target, idxs[:num_to_drop], reg=reg, batch_size=batch_size, num_epochs=num_epochs, verbose=verbose)
