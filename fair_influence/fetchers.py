import pandas as pd

def get_train_test_adult():
  col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
  train = pd.read_csv('./data/adult.data.csv', header=0, names=col_names, index_col=None, skipinitialspace=True)
  test = pd.read_csv('./data/adult.test.csv', header=0, skiprows=1, names=col_names, index_col=None, skipinitialspace=True)
  return train, test
