import pandas as pd
import numpy as np

def prepare_adult(df: pd.DataFrame,
                  protected='race',
                  primary='White') -> np.ndarray:
  col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
  col_names = list(set(df.columns).intersection(set(col_names)))
  cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
  cat_names = list(set(df.columns).intersection(set(cat_names)))
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

def convert_bin(col):
  if len(col) == 0: return col
  default = col[0]
  return np.float32([elem == default for elem in col])

def prepare_saf(df: pd.DataFrame,
                target='frisked',
                protected='race',
                primary='B',
                no_cats=False) -> np.ndarray:
  if isinstance(target, list):
    col_names = [name for name in ['inout', 'recstat','pct', 'ser_num', 'timestop', 'trhsloc', 'perobs', 'crimsusp', 'typeofid', 'explnstp', 'othpers', 'arstoffn', 'sumissue', 'sumoffen', 'offunif', 'officrid', 'contrabn', 'pistol', 'asltweap', 'knifcuti', 'othrweap', 'radio', 'ac_rept', 'ac_inves', 'rf_vcrim', 'rf_othsw', 'ac_proxm', 'rf_attir', 'cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc', 'cs_furtv', 'rf_rfcmp', 'ac_cgdir', 'rf_verbl', 'cs_vcrim', 'cs_bulge', 'cs_other', 'ac_incid', 'ac_time', 'rf_knowl', 'ac_stsnd', 'ac_other', 'sb_hdobj', 'sb_outln', 'sb_admis', 'sb_other', 'rf_furt', 'rf_bulg', 'offverb', 'offshld', 'forceuse', 'sex', 'race', 'age', 'ht_feet', 'weight', 'haircolr', 'eyecolor', 'build', 'city', 'addrpct', 'sector'] if name not in target]
  else:
    col_names = [name for name in ['inout', 'recstat','pct', 'ser_num', 'timestop', 'trhsloc', 'perobs', 'crimsusp', 'typeofid', 'explnstp', 'othpers', 'arstoffn', 'sumissue', 'sumoffen', 'offunif', 'officrid', 'contrabn', 'pistol', 'asltweap', 'knifcuti', 'othrweap', 'radio', 'ac_rept', 'ac_inves', 'rf_vcrim', 'rf_othsw', 'ac_proxm', 'rf_attir', 'cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc', 'cs_furtv', 'rf_rfcmp', 'ac_cgdir', 'rf_verbl', 'cs_vcrim', 'cs_bulge', 'cs_other', 'ac_incid', 'ac_time', 'rf_knowl', 'ac_stsnd', 'ac_other', 'sb_hdobj', 'sb_outln', 'sb_admis', 'sb_other', 'rf_furt', 'rf_bulg', 'offverb', 'offshld', 'forceuse', 'sex', 'race', 'age', 'ht_feet', 'weight', 'haircolr', 'eyecolor', 'build', 'city', 'addrpct', 'sector'] if name != target]
  bin_names = [col_name
               for col_name in ['contrabn','pistol','asltweap','knifcuti','othrweap','radio','ac_rept','ac_inves','rf_vcrim','rf_othsw','ac_proxm','rf_attir','cs_objcs','cs_descr','cs_casng','cs_lkout','rf_vcact','cs_cloth','cs_drgtr','ac_evasv','ac_assoc','cs_furtv','rf_rfcmp','ac_cgdir','rf_verbl','cs_vcrim','cs_bulge','cs_other','ac_incid','ac_time','rf_knowl','ac_stsnd','ac_other','sb_hdobj','sb_outln','sb_admis','sb_other','offunif','sumissue','othpers','explnstp','rf_furt','rf_bulg']
               if col_name not in ['contrabn', 'adtlrept', 'pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap']]
  cat_names = ['pct', 'inout', 'recstat','arstoffn','officrid','trhsloc','crimsusp','typeofid','sumoffen','offverb','offshld','forceuse','sex','race','haircolr','eyecolor','build','city','addrpct','sector']
  not_cat_names = list(set(col_names) - set(cat_names) - set(bin_names))
  except_protected_names = [name for name in cat_names if name != protected]
  not_cat = df[not_cat_names]
  bin_cols = np.vstack([convert_bin(df[col_name]) for col_name in bin_names]).T
  with_dummies = pd.get_dummies(df[except_protected_names], columns=except_protected_names).values
  is_primary = np.float32((df[protected] == primary).values)
  if isinstance(target, list):
    y = 0
    for t in target:
      y += np.float32((df[t].str.contains('Y')).values)
    y = np.float32(y > 0)
  else:
    y = np.float32((df[target].str.contains('Y')).values)
  centered_not_cat = []
  for name in not_cat_names:
    centered_not_cat.append((not_cat[name] - not_cat[name].mean()) / not_cat[name].std())
  if no_cats:
    result = [np.array(centered_not_cat).T,
              bin_cols,
              is_primary[:, np.newaxis]]
  else:
    result = [np.array(centered_not_cat).T,
              bin_cols,
              with_dummies,
              is_primary[:, np.newaxis]]
  data = np.float32(np.concatenate(result, 1))
  return data, y
