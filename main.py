import sys

import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns

import pydash as _

from fair_influence.fetchers import get_train_test_adult, get_train_test_saf
from fair_influence.preprocessing import prepare_adult, prepare_saf
from fair_influence.logistic_regression import TorchLogisticRegression
from fair_influence.fairness import calc_fairness, calc_pred_fairness, make_more_fair_retrain, eval_fairness
from fair_influence.trainer import Trainer
from fair_influence.inference import eval_acc
from fair_influence.influence import calc_log_reg_hvp_inverse, calc_log_reg_grad, calc_s_tests, calc_influences, calc_log_reg_dkl_grad
from fair_influence.explore import scatter_dists, get_fisher_vectors

def adult_analysis():
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
    impacts = make_more_fair_retrain(trainer, X_train, y_train, X_test, reg=0.0, batch_size=1000, num_epochs=1, verbose=False)
    retrain_fair_model = trainer.model
    idxs_to_drop = (impacts > eval_fairness(unfair_model, X_test)).nonzero()[:, 0]
    trainer.retrain_leave_one_out(X_train, y_train, idxs_to_drop, reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
    neg_retrain_fair_model = trainer.model
    print('model loo retrain: KL p(y | female), p(y | male)', eval_fairness(neg_retrain_fair_model, X_test))
    print('acc:', eval_acc(neg_retrain_fair_model, X_test, y_test))
  else:
    calc_hvp_inv = lambda model, grads, reg: calc_log_reg_hvp_inverse(model, X_train, grads, reg=reg)
    calc_grad = lambda model, data, target: calc_log_reg_grad(model, data, target)
    calc_dkl_grad = lambda model, data, target: calc_log_reg_dkl_grad(model, data, target)
    s_tests = calc_s_tests(unfair_model, calc_hvp_inv, calc_dkl_grad, X_test, y_test, reg=0.06)
    influences = calc_influences(unfair_model, calc_grad, s_tests, X_train, y_train).squeeze()
    idxs_to_drop = (influences > 0).nonzero()[:, 0]
    trainer.retrain_leave_one_out(X_train, y_train, idxs_to_drop, reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
    fair_model = trainer.model
    print('model retrain: KL p(y | female), p(y | male)', eval_fairness(fair_model, X_test))
    print('acc:', eval_acc(fair_model, X_test, y_test))
  female = ((X_train[:, -1].byte()).int() & y_train.int()).nonzero().squeeze()
  trainer.retrain_leave_one_out(X_train, y_train, female[:len(idxs_to_drop)], reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
  female_model = trainer.model
  print('female model retrain: KL p(y | female), p(y | male)', eval_fairness(female_model, X_test))
  print('acc:', eval_acc(female_model, X_test, y_test))
  fisher_vectors = get_fisher_vectors(unfair_model, X_train, y_train)
  mask = torch.ones(len(X_train), dtype=torch.uint8)
  mask[idxs_to_drop] = 0
  helpful, hurtful = scatter_dists(fisher_vectors[idxs_to_drop], fisher_vectors[mask])
  plt.close()
  plt.figure()
  # plt.scatter(*helpful.T, c='blue', s=1, label='helpful')
  # plt.scatter(*hurtful.T, c='red', s=1, label='harmful')
  sns.scatterplot(*helpful[torch.empty(len(helpful)).bernoulli(0.1).nonzero().squeeze()].T, label='helpful', palette="Set2", marker='+', s=30)
  sns.scatterplot(*hurtful[torch.empty(len(hurtful)).bernoulli(0.1).nonzero().squeeze()].T, label='hurtful', palette="Set2", marker='x', s=30)
  plt.title('First two principal directions of Fisher embeddings of training points ')
  plt.legend()
  plt.savefig('./fish_pca_hurtful.png')
  # pd.Series(X_train[:, -1][idxs] == y_train[idxs]).rolling(1000).mean().plot()
  # percentage in the protected class AND less than 50k

  remain = raw_train.iloc[list(set(range(len(X_train))) - set(idxs_to_drop.tolist()))]
  remain_X = X_train[list(set(range(len(X_train))) - set(idxs_to_drop.tolist()))]
  remain_y = y_train[list(set(range(len(y_train))) - set(idxs_to_drop.tolist()))]
  orig_corrs = [np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])]
  remain_corrs = [np.corrcoef(remain_X[:, i], remain_y)[0, 1] for i in range(remain_X.shape[1])]

def saf_analysis():
  raw_train, raw_test = get_train_test_saf()
  X, y = [torch.tensor(arr) for arr in prepare_saf(pd.concat([raw_train, raw_test]))]
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
  calc_hvp_inv = lambda model, grads, reg: calc_log_reg_hvp_inverse(model, X_train, grads, reg=reg)
  calc_grad = lambda model, data, target: calc_log_reg_grad(model, data, target)
  calc_dkl_grad = lambda model, data, target: calc_log_reg_dkl_grad(model, data, target)
  s_tests = calc_s_tests(unfair_model, calc_hvp_inv, calc_dkl_grad, X_test, y_test, reg=0.06)
  influences = calc_influences(unfair_model, calc_grad, s_tests, X_train, y_train).squeeze()
  idxs_to_drop = (influences > 0).nonzero()[:, 0]
  trainer.retrain_leave_one_out(X_train, y_train, idxs_to_drop, reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
  fair_model = trainer.model
  print('model retrain: KL p(y | female), p(y | male)', eval_fairness(fair_model, X_test))
  print('acc:', eval_acc(fair_model, X_test, y_test))
  female = ((X_train[:, -1].byte()).int() & y_train.int()).nonzero().squeeze()
  trainer.retrain_leave_one_out(X_train, y_train, female[:len(idxs_to_drop)], reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
  female_model = trainer.model
  print('female model retrain: KL p(y | female), p(y | male)', eval_fairness(female_model, X_test))
  print('acc:', eval_acc(female_model, X_test, y_test))


def main():
  # adult_analysis()
  saf_analysis()

if __name__ == "__main__": main()
