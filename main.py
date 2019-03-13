import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt

from fair_influence.fetchers import get_train_test_adult
from fair_influence.preprocessing import prepare_adult

def main():
  raw_train, raw_test = get_train_test_adult()
  X_train, y_train = prepare_adult(raw_train)
  X_test, y_test = prepare_adult(raw_test)

if __name__ == "__main__": main()
