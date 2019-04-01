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
                primary='W') -> np.ndarray:
  assert target in ['frisked', 'searched']
  col_names = ['inout', 'recstat','pct', 'ser_num', 'timestop', 'trhsloc', 'perobs', 'crimsusp', 'typeofid', 'explnstp', 'othpers', 'arstmade', 'arstoffn', 'sumissue', 'sumoffen', 'offunif', 'officrid', 'contrabn', 'adtlrept', 'pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap', 'pf_hands', 'pf_wall', 'pf_grnd', 'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_pepsp', 'pf_other', 'radio', 'ac_rept', 'ac_inves', 'rf_vcrim', 'rf_othsw', 'ac_proxm', 'rf_attir', 'cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc', 'cs_furtv', 'rf_rfcmp', 'ac_cgdir', 'rf_verbl', 'cs_vcrim', 'cs_bulge', 'cs_other', 'ac_incid', 'ac_time', 'rf_knowl', 'ac_stsnd', 'ac_other', 'sb_hdobj', 'sb_outln', 'sb_admis', 'sb_other', 'repcmd', 'revcmd', 'rf_furt', 'rf_bulg', 'offverb', 'offshld', 'forceuse', 'sex', 'race', 'age', 'ht_feet', 'weight', 'haircolr', 'eyecolor', 'build', 'city', 'addrpct', 'sector']
  bin_names = ['contrabn','adtlrept','pistol','riflshot','asltweap','knifcuti','machgun','othrweap','pf_hands','pf_wall','pf_grnd','pf_drwep','pf_ptwep','pf_baton','pf_hcuff','pf_pepsp','pf_other','radio','ac_rept','ac_inves','rf_vcrim','rf_othsw','ac_proxm','rf_attir','cs_objcs','cs_descr','cs_casng','cs_lkout','rf_vcact','cs_cloth','cs_drgtr','ac_evasv','ac_assoc','cs_furtv','rf_rfcmp','ac_cgdir','rf_verbl','cs_vcrim','cs_bulge','cs_other','ac_incid','ac_time','rf_knowl','ac_stsnd','ac_other','sb_hdobj','sb_outln','sb_admis','sb_other','offunif','sumissue','othpers','arstmade','explnstp','rf_furt','rf_bulg']
  cat_names = ['inout', 'recstat','arstoffn','officrid','trhsloc','crimsusp','typeofid','sumoffen','offverb','offshld','forceuse','sex','race','age','ht_feet','weight','haircolr','eyecolor','build','city','addrpct','sector']
  not_cat_names = list(set(col_names) - set(cat_names) - set(bin_names))
  except_protected_names = [name for name in cat_names if name != protected]
  not_cat = df[not_cat_names]
  bin_cols = np.vstack([convert_bin(df[col_name]) for col_name in bin_names]).T
  with_dummies = pd.get_dummies(df[except_protected_names], columns=except_protected_names).values
  is_primary = np.float32((df[protected] == primary).values)
  y = np.float32((df[target].str.contains('Y')).values)
  centered_not_cat = []
  for name in not_cat_names:
    centered_not_cat.append((not_cat[name] - not_cat[name].mean()) / not_cat[name].std())
  data = np.float32(np.concatenate([np.array(centered_not_cat).T,
                                    bin_cols,
                                    with_dummies,
                                    is_primary[:, np.newaxis]],
                                   1))
  return data, y
