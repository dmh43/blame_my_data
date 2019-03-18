import torch

from .fairness import calc_pred_fairness

def eval_fairness(model, data):
  model.eval()
  with torch.no_grad():
    preds = (model(data) > 0.5)
  return calc_pred_fairness(data, preds)

def eval_acc(model, data, target):
  model.eval()
  with torch.no_grad():
    preds = (model(data) > 0.5)
  return torch.sum(preds.float() == target).float() / len(target)
