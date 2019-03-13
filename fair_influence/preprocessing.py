import pandas as pd
import numpy as np

def prepare_adult(df: pd.DataFrame,
                  protected='race',
                  primary='white') -> np.ndarray:
  col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
  cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
  not_cat_names = list(set(col_names) - set(cat_names))
  except_protected_names = [name for name in cat_names if name != protected]
  not_cat = df[not_cat_names]
  with_dummies = pd.get_dummies(df[except_protected_names], columns=except_protected_names).values
  is_primary = (df[protected] == primary).values
  y = (df['target'].str.contains('>')).values
  return np.concatenate([not_cat, with_dummies, is_primary[:, np.newaxis]], 1), y
