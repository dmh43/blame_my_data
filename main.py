import sys

import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

import pydash as _

from fair_influence.fetchers import get_train_test_adult
from fair_influence.preprocessing import prepare_adult
from fair_influence.logistic_regression import TorchLogisticRegression
from fair_influence.fairness import calc_fairness, calc_pred_fairness, make_more_fair_retrain, eval_fairness
from fair_influence.trainer import Trainer
from fair_influence.inference import eval_acc
from fair_influence.influence import calc_log_reg_hvp_inverse, calc_log_reg_grad, calc_s_tests, calc_influences, calc_log_reg_dkl_grad
from fair_influence.explore import scatter_dists, get_fisher_vectors

def main():
  raw_train, raw_test = get_train_test_adult()
  raw_train = raw_train[['age', 'fnlwgt', 'education-num', 'race', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'target']]
  raw_test = raw_test[['age', 'fnlwgt', 'education-num', 'race', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'target']]
  X, y = [torch.tensor(arr) for arr in prepare_adult(pd.concat([raw_train, raw_test]), protected='sex', primary='Male')]
  X_train = X[:len(raw_train)]
  y_train = y[:len(raw_train)]
  X_test  = X[len(raw_train):len(raw_train) + len(raw_test)]
  y_test  = y[len(raw_train):len(raw_train) + len(raw_test)]
  get_model = lambda: TorchLogisticRegression(X_train.shape[1])
  trainer = Trainer(get_model, nn.BCELoss())
  trainer.train(X_train, y_train, batch_size=1000, num_epochs=100, reg=0.0, verbose=False)
  unfair_model = trainer.model
  print('acc:', eval_acc(unfair_model, X_test, y_test))
  print('train: KL p(y | female), p(y | male)', calc_fairness(X_train, y_train))
  print('test: KL p(y | female), p(y | male)', calc_fairness(X_test, y_test))
  print('model: KL p(y | female), p(y | male)', eval_fairness(unfair_model, X_test))
  if '--retrain' in sys.argv:
    impacts = make_more_fair_retrain(trainer, X_train, y_train, X_test)
    fair_model = trainer.model
  else:
    calc_hvp_inv = lambda model, grads, reg: calc_log_reg_hvp_inverse(model, X_train, grads, reg=reg)
    calc_grad = lambda model, data, target: calc_log_reg_grad(model, data, target)
    calc_dkl_grad = lambda model, data, target: calc_log_reg_dkl_grad(model, data, target)
    s_tests = calc_s_tests(unfair_model, calc_hvp_inv, calc_dkl_grad, X_test, y_test, reg=0.06)
    influences = calc_influences(unfair_model, calc_grad, s_tests, X_train, y_train)
    idxs_to_drop = (influences > 0).nonzero()[:, 0]
    trainer.retrain_leave_one_out(X_train, y_train, idxs_to_drop, reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
    fair_model = trainer.model
    print('model retrain: KL p(y | female), p(y | male)', eval_fairness(fair_model, X_test))
    print('acc:', eval_acc(fair_model, X_test, y_test))
  female = (X_train[:, -1].int() & y_train.int()).nonzero().squeeze()
  trainer.retrain_leave_one_out(X_train, y_train, female[:len(idxs_to_drop)], reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
  female_model = trainer.model
  print('female model retrain: KL p(y | female), p(y | male)', eval_fairness(female_model, X_test))
  print('acc:', eval_acc(female_model, X_test, y_test))
  fisher_vectors = get_fisher_vectors(unfair_model, X_train, y_train)
  mask = torch.ones(len(X_train), dtype=torch.uint8)
  mask[idxs_to_drop] = 0
  helpful, hurtful = scatter_dists(fisher_vectors[idxs_to_drop], fisher_vectors[mask])
  plt.scatter(*helpful.T, c='blue', s=10)
  plt.scatter(*hurtful.T, c='red', s=10)
  # pd.Series(X_train[:, -1][idxs] == y_train[idxs]).rolling(1000).mean().plot()
  # percentage in the protected class AND less than 50k

if __name__ == "__main__": main()
