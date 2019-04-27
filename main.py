import sys
from operator import itemgetter

import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
from random import seed, sample

from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import tree
import graphviz

import pydash as _

from fair_influence.fetchers import get_train_test_adult, get_train_test_saf
from fair_influence.preprocessing import prepare_adult, prepare_saf
from fair_influence.logistic_regression import TorchLogisticRegression
from fair_influence.fairness import calc_fairness, calc_pred_fairness, make_more_fair_retrain, eval_fairness
from fair_influence.trainer import Trainer
from fair_influence.inference import eval_acc
from fair_influence.influence import calc_log_reg_hvp_inverse, calc_log_reg_grad, calc_s_tests, calc_influences, calc_log_reg_dkl_grad
from fair_influence.explore import scatter_dists, get_fisher_vectors
from fair_influence.helpers import colgetter

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
  seed(1)
  raw_train, raw_test = get_train_test_saf()
  data = pd.concat([raw_train, raw_test])
  X, y = [torch.tensor(arr) for arr in prepare_saf(data,
                                                   target='frisked',
                                                   protected='race',
                                                   primary='W')]

  # X_train = X[:len(raw_train)]
  # y_train = y[:len(raw_train)]
  # X_test  = X[len(raw_train):len(raw_train) + len(raw_test)]
  # y_test  = y[len(raw_train):len(raw_train) + len(raw_test)]

  pct = ~(raw_train.pct == 40)
  # pct = raw_train.pct < 70
  not_pct = ~pct
  pct = pct.nonzero()[0]
  not_pct = not_pct.nonzero()[0]
  X_train = X[pct]
  y_train = y[pct]
  X_test  = X[not_pct]
  y_test  = y[not_pct]
  print('train: KL p(y | nonwhite), p(y | white)', calc_fairness(X_train, y_train))
  print('test: KL p(y | nonwhite), p(y | white)', calc_fairness(X_test, y_test))
  get_model = lambda: TorchLogisticRegression(X_train.shape[1])
  trainer = Trainer(get_model, nn.BCELoss())
  trainer.train(X_train, y_train, batch_size=1000, num_epochs=100, reg=0.0, verbose=False)
  unfair_model = trainer.model
  print('acc:', eval_acc(unfair_model, X_test, y_test))
  print('model: KL p(y | nonwhite), p(y | white)', eval_fairness(unfair_model, X_train))
  print('model: KL p(y | nonwhite), p(y | white)', eval_fairness(unfair_model, X_test))
  calc_hvp_inv = lambda model, grads, reg: calc_log_reg_hvp_inverse(model, X_train, grads, reg=reg)
  s_tests = calc_s_tests(unfair_model,
                         calc_hvp_inv,
                         calc_log_reg_dkl_grad,
                         X_test,
                         y_test,
                         reg=0.05)
  influences = calc_influences(unfair_model,
                               calc_log_reg_grad,
                               s_tests,
                               X_train,
                               y_train).squeeze()
  idxs_to_drop = (influences > 0).nonzero()[:, 0]
  not_idxs_to_drop = (influences <= 0).nonzero()[:, 0]
  trainer.retrain_leave_one_out(X_train, y_train, idxs_to_drop, reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
  fair_model = trainer.model
  print('model retrain: KL p(y | nonwhite), p(y | white)', eval_fairness(fair_model, X_train))
  print('model retrain: KL p(y | nonwhite), p(y | white)', eval_fairness(fair_model, X_test))
  print('acc:', eval_acc(fair_model, X_test, y_test))
  nonwhite = ((X_train[:, -1].byte()).int() & y_train.int()).nonzero().squeeze()
  trainer.retrain_leave_one_out(X_train, y_train, nonwhite[:len(idxs_to_drop)], reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
  nonwhite_model = trainer.model
  print('nonwhite model retrain: KL p(y | nonwhite), p(y | white)', eval_fairness(nonwhite_model, X_train))
  print('nonwhite model retrain: KL p(y | nonwhite), p(y | white)', eval_fairness(nonwhite_model, X_test))
  print('acc:', eval_acc(nonwhite_model, X_test, y_test))

  infs, idxs = torch.sort(influences, descending=True)
  raw_train.iloc[idxs[:100]]

def saf_provenance_analysis():
  seed(1)
  raw_train, raw_test = get_train_test_saf()
  data = pd.concat([raw_train, raw_test])
  X, y = [torch.tensor(arr) for arr in prepare_saf(data,
                                                   target=['contrabn', 'adtlrept', 'pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap'],
                                                   protected='race',
                                                   primary='W',
                                                   no_cats=True)]
  get_model = lambda: TorchLogisticRegression(X.shape[1])
  trainer = Trainer(get_model, nn.BCELoss())
  trainer.train(X, y, batch_size=1000, num_epochs=100, reg=0.0, verbose=False)
  unfair_model = trainer.model
  print('acc:', eval_acc(unfair_model, X, y))
  print('model: KL p(y | nonwhite), p(y | white)', eval_fairness(unfair_model, X))
  influences_by_pct = {}
  pct_nums = data.pct.unique()
  for pct_num in pct_nums:
    pct = raw_train.pct == pct_num
    not_pct = ~pct
    pct = pct.nonzero()[0]
    not_pct = not_pct.nonzero()[0]
    X_train_pct = X[pct]
    y_train_pct = y[pct]
    X_test_pct  = X[not_pct]
    y_test_pct  = y[not_pct]
    calc_hvp_inv = lambda model, grads, reg, X_train_pct=X_train_pct: calc_log_reg_hvp_inverse(model, X_train_pct, grads, reg=reg)
    s_tests = calc_s_tests(unfair_model,
                           calc_hvp_inv,
                           calc_log_reg_dkl_grad,
                           X_test_pct,
                           y_test_pct,
                           reg=0.05)
    influences = calc_influences(unfair_model,
                                 calc_log_reg_grad,
                                 s_tests,
                                 X_train_pct,
                                 y_train_pct).squeeze()
    pct_influence = influences.sum()
    influences_by_pct[pct_num] = pct_influence.item()

  frisk_white_by_pct = {pct_num: ((data.race[data.pct == pct_num] == 'W') & y[(data.pct == pct_num).nonzero()].byte()).mean() for pct_num in pct_nums}
  pct_ranked_by_influence = sorted(influences_by_pct.items(), key=itemgetter(1))
  pct_ranked_by_frisk_white = sorted(frisk_white_by_pct.items(), key=itemgetter(1))
  print(kendalltau(*map(colgetter(0), (pct_ranked_by_influence, pct_ranked_by_frisk_white))))

  for r in ['W', 'B']:
    print(r, 'y overall', ((data.race==r) & y.byte()).values.sum() / (data.race==r).values.sum())
    print(r, 'not y overall', ((data.race==r) & pd.Series(~y.byte())).values.sum() / (data.race==r).values.sum())
    pct = 106
    print(r, f'y pct {pct}',
          ((data.pct == pct) & (data.race==r) & pd.Series(y.byte())).values.sum() / ((data.pct == pct) & (data.race==r)).values.sum())
    print(r, 'not y pct 106',
          ((data.pct == pct) & (data.race==r) & pd.Series(~y.byte())).values.sum() / ((data.pct == pct) & (data.race==r)).values.sum())

  calc_hvp_inv = lambda model, grads, reg, X=X: calc_log_reg_hvp_inverse(model, X, grads, reg=reg)
  s_tests = calc_s_tests(unfair_model,
                         calc_hvp_inv,
                         calc_log_reg_dkl_grad,
                         X,
                         y,
                         reg=0.05)
  influences = calc_influences(unfair_model,
                               calc_log_reg_grad,
                               s_tests,
                               X,
                               y).squeeze()

  # fisher_vectors = get_fisher_vectors(unfair_model, X, y)
  model = DecisionTreeClassifier(max_depth=3)
  # model.fit(fisher_vectors, influences > 0)
  model.fit(np.concatenate([X, y[:, np.newaxis]], 1), influences > 0)
  dot_data = tree.export_graphviz(model, out_file=None)
  graph = graphviz.Source(dot_data)
  graph.render("iris")

  plt.hist(influences[X[:, -1].nonzero()].squeeze(), bins=100, label='W', density=True)
  plt.hist(influences[(~X[:, -1].byte()).nonzero()].squeeze(), bins=100, label='~W', density=True)
  plt.legend()
  plt.show()

  corrs = {}
  for col_name in data.columns:
    if col_name == 'race':
      corr, p = spearmanr(data[col_name] == 'W', influences)
      corrs[col_name] = 0.0 if np.isnan(corr) else corr
    try:
      # corrs[col_name] = pearsonr(data[col_name] == 'Y', data.race == 'W')[0]
      corr, p = spearmanr(data[col_name] == 'Y', influences)
      corrs[col_name] = 0.0 if np.isnan(corr) else corr
    except:
      pass
  cols_by_corr = sorted(corrs.items(), key=itemgetter(1))

def saf_active_learning():
  seed(1)
  raw_train, raw_test = get_train_test_saf()
  data = pd.concat([raw_train, raw_test])
  X, y = [torch.tensor(arr) for arr in prepare_saf(data,
                                                   target=['contrabn', 'adtlrept', 'pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap'],
                                                   protected='race',
                                                   primary='W',
                                                   no_cats=True)]

  for num in [1000, 800, 500, 10]:
    first_holdout_idx = int(X[:, -1].nonzero()[-num])
    white_holdout = X[:, -1].nonzero()[-num:].squeeze()
    all_holdout = np.array(sample(range(first_holdout_idx, len(X)), len(white_holdout)))
    for method, holdout_idxs in (('white', white_holdout), ('all', all_holdout)):
      X_holdout = X[holdout_idxs]
      y_holdout = y[holdout_idxs]
      keep_idxs = np.array(list(set(range(len(X))) - set(holdout_idxs.tolist())))
      X_sampled = X[keep_idxs]
      y_sampled = y[keep_idxs]

      get_model = lambda: TorchLogisticRegression(X_sampled.shape[1])
      trainer = Trainer(get_model, nn.BCELoss())
      trainer.train(X_sampled, y_sampled, batch_size=1000, num_epochs=100, reg=0.0, verbose=False)
      unfair_model = trainer.model
      print('sampling', method)
      print('num W withheld', num)
      print('acc:', eval_acc(unfair_model, X_sampled, y_sampled))
      print('model: KL p(y | nonwhite), p(y | white)', eval_fairness(unfair_model, X_sampled), np.log(eval_fairness(unfair_model, X_sampled)))

      calc_hvp_inv = lambda model, grads, reg, X_sampled=X_sampled: calc_log_reg_hvp_inverse(model, X_sampled, grads, reg=reg)
      s_tests = calc_s_tests(unfair_model,
                             calc_hvp_inv,
                             calc_log_reg_dkl_grad,
                             X_sampled,
                             y_sampled,
                             reg=0.05)
      influences = calc_influences(unfair_model,
                                   calc_log_reg_grad,
                                   s_tests,
                                   X_sampled,
                                   y_sampled).squeeze()

      plt.figure()
      plt.hist(np.log(np.abs(influences[X_sampled[:, -1].nonzero()].squeeze())), bins=100, label='W', density=True)
      plt.hist(np.log(np.abs(influences[(~X_sampled[:, -1].byte()).nonzero()].squeeze())), bins=100, label='~W', density=True)
      plt.legend()
      plt.savefig('withhold_' + str(num) + method + '.png')
  plt.close()

  calc_hvp_inv = lambda model, grads, reg, X=X: calc_log_reg_hvp_inverse(model, X, grads, reg=reg)
  s_tests = calc_s_tests(unfair_model,
                         calc_hvp_inv,
                         calc_log_reg_dkl_grad,
                         X,
                         y,
                         reg=0.05)
  influences = calc_influences(unfair_model,
                               calc_log_reg_grad,
                               s_tests,
                               X,
                               y).squeeze()

  # idxs_last_mode = (np.log(influences[X[:, -1].nonzero()].squeeze()) > -3).nonzero().squeeze()
  # idxs_not_last_mode = (~(np.log(influences[X[:, -1].nonzero()].squeeze()) > -3)).nonzero().squeeze()
  # X_last_mode = np.concatenate([np.concatenate([X[idxs_last_mode], X[idxs_not_last_mode]])], 1)
  # y_last_mode = np.zeros(X[:, -1].sum().int())
  # y_last_mode[:len(idxs_last_mode)] = 1
  # pca = PCA(n_components=2, whiten=True)
  # plt.scatter(*pca.fit_transform(X_last_mode).T, c=((data.frisked == 'Y')[(data.race == 'W')]))
  # model = DecisionTreeClassifier(max_depth=3)
  # model = LogisticRegression()
  # model.fit(X_last_mode, y_last_mode)
  # dot_data = tree.export_graphviz(model, out_file=None)
  # graph = graphviz.Source(dot_data)
  # graph.render("last_mode")

def saf_predict_inf():
  seed(1)
  raw_train, raw_test = get_train_test_saf()
  data = pd.concat([raw_train, raw_test])
  X, y = [torch.tensor(arr) for arr in prepare_saf(data,
                                                   target=['contrabn', 'adtlrept', 'pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap'],
                                                   protected='race',
                                                   primary='W',
                                                   no_cats=True)]

  # X_train = X[:len(raw_train)]
  # y_train = y[:len(raw_train)]
  # X_test  = X[len(raw_train):len(raw_train) + len(raw_test)]
  # y_test  = y[len(raw_train):len(raw_train) + len(raw_test)]

  infs = []
  for pct in data.pct.unique():
    infs.append((pct, calc_fairness(X[(raw_train.pct == pct).nonzero()],
                                    y[(raw_train.pct == pct).nonzero()]).item()))

  print(sorted(infs, key=itemgetter(1)))

  pct = ~(raw_train.pct == 13)
  # pct = raw_train.pct < 70
  not_pct = ~pct
  pct = pct.nonzero()[0]
  not_pct = not_pct.nonzero()[0]
  X_train = X[pct]
  y_train = y[pct]
  X_test  = X[not_pct]
  y_test  = y[not_pct]
  print('train: KL p(y | nonwhite), p(y | white)', calc_fairness(X_train, y_train))
  print('test: KL p(y | nonwhite), p(y | white)', calc_fairness(X_test, y_test))
  get_model = lambda: TorchLogisticRegression(X_train.shape[1])
  trainer = Trainer(get_model, nn.BCELoss())
  trainer.train(X_train, y_train, batch_size=1000, num_epochs=100, reg=0.0, verbose=False)
  unfair_model = trainer.model
  print('acc:', eval_acc(unfair_model, X_test, y_test))
  print('model: KL p(y | nonwhite), p(y | white)', eval_fairness(unfair_model, X_train))
  print('model: KL p(y | nonwhite), p(y | white)', eval_fairness(unfair_model, X_test))
  calc_hvp_inv = lambda model, grads, reg: calc_log_reg_hvp_inverse(model, X_train, grads, reg=reg)
  s_tests = calc_s_tests(unfair_model,
                         calc_hvp_inv,
                         calc_log_reg_dkl_grad,
                         X_test,
                         y_test,
                         reg=0.05)
  influences = calc_influences(unfair_model,
                               calc_log_reg_grad,
                               s_tests,
                               X_train,
                               y_train).squeeze()
  idxs_to_drop = (influences > 0).nonzero()[:, 0]
  not_idxs_to_drop = (influences <= 0).nonzero()[:, 0]
  trainer.retrain_leave_one_out(X_train, y_train, idxs_to_drop, reg=0.0, batch_size=1000, num_epochs=100, verbose=False)
  fair_model = trainer.model
  print('model retrain: KL p(y | nonwhite), p(y | white)', eval_fairness(fair_model, X_train))
  print('model retrain: KL p(y | nonwhite), p(y | white)', eval_fairness(fair_model, X_test))
  print('acc:', eval_acc(fair_model, X_test, y_test))

  model = DecisionTreeClassifier(max_depth=3)
  model.fit(X_train, influences > 0)
  dot_data = tree.export_graphviz(model, out_file=None)
  graph = graphviz.Source(dot_data)
  graph.render("pred_inf")

  model = LogisticRegression()
  model.fit(X_train, influences > 0)


def main():
  # adult_analysis()
  saf_analysis()

if __name__ == "__main__": main()
