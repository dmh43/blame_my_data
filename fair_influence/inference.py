import torch

def eval_acc(model, data, target):
  model.eval()
  with torch.no_grad():
    preds = (model(data) > 0.5)
  return torch.sum(preds.float() == target).float() / len(target)
