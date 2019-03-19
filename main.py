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
from fair_influence.logistic_regression import LogisticRegression
from fair_influence.fairness import calc_fairness, calc_pred_fairness, make_more_fair_retrain, eval_fairness
from fair_influence.trainer import Trainer
from fair_influence.inference import eval_acc
from fair_influence.influence import calc_log_reg_hvp_inverse, calc_log_reg_grad, calc_s_tests, calc_influences, calc_log_reg_dkl_grad

def main():
  raw_train, raw_test = get_train_test_adult()
  X, y = [torch.tensor(arr) for arr in prepare_adult(pd.concat([raw_train, raw_test]))]
  X_train = X[:len(raw_train)]
  y_train = y[:len(raw_train)]
  X_test  = X[:len(raw_test)]
  y_test  = y[:len(raw_test)]
  get_model = lambda: LogisticRegression(X_train.shape[1])
  trainer = Trainer(get_model, nn.BCELoss())
  trainer.train(X_train, y_train, batch_size=1000, num_epochs=30, reg=0.1)
  print('acc:', eval_acc(trainer.model, X_test, y_test))
  print('train: KL p(y | white), p(y | nonwhite)', calc_fairness(X_train, y_train))
  print('test: KL p(y | white), p(y | nonwhite)', calc_fairness(X_test, y_test))
  print('model: KL p(y | white), p(y | nonwhite)', eval_fairness(trainer.model, X_test))
  if '--retrain' in sys.argv:
    make_more_fair_retrain(trainer, X_train, y_train, X_test)
    print('model retrain: KL p(y | white), p(y | nonwhite)', eval_fairness(trainer.model, X_test))
  else:
    calc_hvp_inv = lambda grads: calc_log_reg_hvp_inverse(trainer.model, X_train, grads)
    calc_grad = lambda data, target: calc_log_reg_grad(trainer.model, data, target)
    calc_dkl_grad = lambda data, target: calc_log_reg_dkl_grad(trainer.model, data, target)
    s_tests = calc_s_tests(calc_hvp_inv, calc_dkl_grad, X_test, y_test)
    influences = calc_influences(calc_grad, s_tests, X_train, y_train)
    infs, idxs = torch.sort(influences.squeeze(), descending=True)
    trainer.retrain_leave_one_out(X_train, y_train, idxs[:100], reg=0.1, batch_size=1000, num_epochs=30, verbose=True)
    print('model: KL p(y | white), p(y | nonwhite)', eval_fairness(trainer.model, X_test))

if __name__ == "__main__": main()
