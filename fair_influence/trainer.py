from typing import Callable

import torch
from torch.optim import Adam

from .fairness import calc_pred_fairness

class Trainer():
  def __init__(self, get_model, criteria):
    self._get_model = get_model
    self._init_model_and_opt()
    self.criteria = criteria

  def _init_model_and_opt(self):
    self.model = self._get_model()
    self.optimizer = Adam(self.params)

  @property
  def params(self):
    return list(self.model.parameters())

  def train(self, data, target, reg=0.0, batch_size=1000, num_epochs=10, verbose=True):
    for epoch_num in range(num_epochs):
      if verbose: print('Epoch', epoch_num)
      idxs = torch.randperm(len(data))
      for batch_num, batch_idxs in enumerate(torch.chunk(idxs, len(data) // batch_size + 1)):
        batch_data = data[batch_idxs]
        batch_target = target[batch_idxs]
        self._batch_train(batch_data, batch_target, reg, batch_num, verbose)

  def assess_influence_retrain(self,
                               data,
                               target,
                               reg=0.0,
                               lim=None,
                               batch_size=1000,
                               num_epochs=10,
                               verbose=True):
    fairness = []
    for idx in range(min(len(data), lim) if lim is not None else len(data)):
      self._retrain_leave_one_out(data, target, idx, reg, batch_size, num_epochs, verbose)
      self.model.eval()
      preds = self.model(data) > 0.5
      fairness.append(calc_pred_fairness(data, preds).item())
    return torch.tensor(fairness)

  def _retrain_leave_one_out(self, data, target, idx_to_leave_out, reg, batch_size, num_epochs, verbose):
    self._init_model_and_opt()
    mask = torch.ones(len(data), dtype=torch.uint8)
    mask[idx_to_leave_out] = 0
    self.train(data[mask], target[mask], reg, batch_size=batch_size, num_epochs=num_epochs, verbose=verbose)

  def _batch_train(self, batch_data, batch_target, reg, batch_num, verbose):
    self.optimizer.zero_grad()
    out = self.model(batch_data)
    loss = self.criteria(out, batch_target)
    for param in self.model.parameters():
      loss += torch.norm(param) * reg
    if verbose and (batch_num % 10) == 0: print('batch num:', batch_num, 'loss:', loss)
    loss.backward()
    self.optimizer.step()
