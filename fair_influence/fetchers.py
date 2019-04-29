from random import shuffle

import pandas as pd
import numpy as np
import numpy.random as rn

def get_train_test_synth(n, train_size=0.8):
  A = rn.randint(0, 2, size=n)
  U = rn.normal(loc=0.0, scale=1.0, size=n)
  R = np.float32((A * (U > 0)))
  R[int(n * train_size):] = A[int(n * train_size):]
  X = U[:, np.newaxis] + rn.normal(loc=0.0, scale=1.0, size=(n, 4))
  y = np.float32(U + rn.normal(loc=1.0, scale=0.1, size=n) > 0)
  data = np.float32(np.concatenate([X, R[:, np.newaxis], A[:, np.newaxis]], 1))
  return data[:int(n * train_size)], y[:int(n * train_size)], data[int(n * train_size):], y[int(n * train_size):]

def get_train_test_adult():
  col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
  train = pd.read_csv('./data/adult.data.csv', header=0, names=col_names, index_col=None, skipinitialspace=True)
  test = pd.read_csv('./data/adult.test.csv', header=0, skiprows=1, names=col_names, index_col=None, skipinitialspace=True)
  return train, test

def get_train_test_saf(do_shuffle=False):
  saf = pd.read_csv('./data/2016_sqf_database.csv')
  saf = saf[[col
             for col in saf.columns
             if col not in ['perstop']]]
  saf.age = saf.age.map(lambda val: 20 if val == '**' else int(val))
  idxs = list(range(len(saf)))
  if do_shuffle: shuffle(idxs)
  train = saf.iloc[idxs[:int(0.8 * len(saf))]]
  test = saf.iloc[idxs[int(0.8 * len(saf)):]]
  return train, test
