# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from sklearn.metrics import roc_auc_score
from sklearn.base import clone

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.lmdd import LMDD
from pyod.utils.data import generate_data


class TestCOF(unittest.TestCase):
    def setUp(self):
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.6
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = LMDD(contamination=self.contamination, random_state=42)
        self.clf.fit(self.X_train)

    def test_sklearn_estimator(self):
        # check_estimator(self.clf)
        pass

    def test_parameters(self):
        if not (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None):
            raise AssertionError
        if not (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None):
            raise AssertionError
        if not (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None):
            raise AssertionError
        if not (hasattr(self.clf, 'dis_measure_') and
                self.clf.dis_measure_ is not None):
            raise AssertionError
        if not (hasattr(self.clf, 'n_iter_') and
                self.clf.n_iter_ is not None):
            raise AssertionError
        if not (hasattr(self.clf, 'random_state_') and
                self.clf.random_state_ is not None):
            raise AssertionError

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        if (roc_auc_score(self.y_test, pred_scores) < self.roc_floor):
            raise AssertionError

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        if (pred_proba.min() < 0):
            raise AssertionError
        if (pred_proba.max() > 1):
            raise AssertionError

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        if (pred_proba.min() < 0):
            raise AssertionError
        if (pred_proba.max() > 1):
            raise AssertionError

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        if (pred_proba.min() < 0):
            raise AssertionError
        if (pred_proba.max() > 1):
            raise AssertionError

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        if (confidence.min() < 0):
            raise AssertionError
        if (confidence.max() > 1):
            raise AssertionError

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(self.X_test,
                                                        method='linear',
                                                        return_confidence=True)
        if (pred_proba.min() < 0):
            raise AssertionError
        if (pred_proba.max() > 1):
            raise AssertionError

        assert_equal(confidence.shape, self.y_test.shape)
        if (confidence.min() < 0):
            raise AssertionError
        if (confidence.max() > 1):
            raise AssertionError

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    @staticmethod
    def test_check_parameters():
        with assert_raises(ValueError):
            LMDD(contamination=10.)
        with assert_raises(ValueError):
            LMDD(dis_measure='unknown')
        with assert_raises(TypeError):
            LMDD(dis_measure=5)
        with assert_raises(TypeError):
            LMDD(n_iter='not int')
        with assert_raises(ValueError):
            LMDD(n_iter=-1)
        with assert_raises(ValueError):
            LMDD(random_state='not valid')
        with assert_raises(ValueError):
            LMDD(random_state=-1)

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
