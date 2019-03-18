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
from fair_influence.fairness import kl_div, calc_dist_cond_protected, calc_fairness, calc_pred_fairness
from fair_influence.trainer import Trainer

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
  model = trainer.model
  model.eval()
  with torch.no_grad():
    preds = (model(X_test) > 0.5)
    print('acc:', torch.sum(preds.float() == y_test).float() / len(y_test))
  print('train: KL p(y | white), p(y | nonwhite)', calc_fairness(X_train, y_train))
  print('test: KL p(y | white), p(y | nonwhite)', calc_fairness(X_test, y_test))
  print('model: KL p(y | white), p(y | nonwhite)', calc_pred_fairness(X_test, preds))
  print('retrain fairness', trainer.assess_influence_retrain(X_train,
                                                             y_train,
                                                             reg=0.1,
                                                             batch_size=1000,
                                                             num_epochs=30,
                                                             lim=10,
                                                             verbose=False))

if __name__ == "__main__": main()
