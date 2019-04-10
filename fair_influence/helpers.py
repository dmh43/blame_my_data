def colgetter(col_num):
  return lambda coll: list(zip(*coll))[col_num]
