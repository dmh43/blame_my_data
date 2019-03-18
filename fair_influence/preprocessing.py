import pandas as pd
import numpy as np

def prepare_adult(df: pd.DataFrame,
                  protected='race',
                  primary='White') -> np.ndarray:
  col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
  cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
  not_cat_names = list(set(col_names) - set(cat_names))
  except_protected_names = [name for name in cat_names if name != protected]
  not_cat = df[not_cat_names]
  with_dummies = pd.get_dummies(df[except_protected_names], columns=except_protected_names).values
  is_primary = (df[protected] == primary).values
  y = np.float32((df['target'].str.contains('>')).values)
  centered_not_cat = []
  for name in not_cat_names:
    centered_not_cat.append((not_cat[name] - not_cat[name].mean()) / not_cat[name].std())
  data = np.float32(np.concatenate([np.array(centered_not_cat).T,
                                    with_dummies,
                                    is_primary[:, np.newaxis]],
                                   1))
  return data, y
