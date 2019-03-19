    :Author: Dany Haddad

.. contents::

1 Dont blame me blame my data
-----------------------------

Identify a subset of the training data that makes a classifier
unfair. In this demo, we use Demographic Parity as our fairness
constraint, so unless :math:`p(\hat{Y} | A = a) = p(\hat{Y} | A = a')
\forall a, a'`, then we declare our classifier as unfair. For
simplicity, we first consider the case with only a single binary
protected attribute.

We evaluate the distance between these two distributions using the KL
divergence: 

.. math::

    D_{KL} (P || Q) = \sum_y P(y) \log(\frac{P(y)}{Q(y)})


where :math:`P(y) = p(\hat{Y} = y | A = 0)` and :math:`Q(y) = p(\hat{Y} = y | A =
1)` were :math:`A = 0` indicates the protected group. We ues the KL
divergence as our metric for unfairness, the larger the distance
between these two distributions, the more unfair our classifier is.

We take two approaches to determining how each training example
affects our fairness on a test set.

1. Leave-one-out retraining of the model for each training point and evaluating the change in unfairness for each model.

2. Computing the influence of each training point on the unfairness
   over the test set. We follow the approach of `Koh and Liang <https://arxiv.org/pdf/1703.04730.pdf>`_,
   extending it to the case where we evaluate influence of a training
   point on a criteria different from the loss function the model was
   trained on.

After evaluating the impact of each training point on our classifier's
fairness, we select a subset that is most responsible for the
unfairness exhibited on the test set. Similar to `Khanna et al. <https://arxiv.org/pdf/1810.10118.pdf>`_, we
take advantage of the submodularity of this objective and greedily
insert training examples into this set.
