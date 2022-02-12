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

from pyod.models.deep_svdd import DeepSVDD
from pyod.utils.data import generate_data


class TestDeepSVDD(unittest.TestCase):
    def setUp(self):
        self.n_train = 6000
        self.n_test = 1000
        self.n_features = 300
        self.contamination = 0.1
        self.roc_floor = 0.5
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = DeepSVDD(epochs=10, hidden_neurons=[64, 32],
                            contamination=self.contamination, random_state=2021)
        self.clf_ae = DeepSVDD(epochs=5, use_ae=True, output_activation='relu',
                               hidden_neurons=[16, 8, 4], contamination=self.contamination,
                               preprocessing=False)
        self.clf.fit(self.X_train)
        self.clf_ae.fit(self.X_train)

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
        if not (hasattr(self.clf, '_mu') and
                self.clf._mu is not None):
            raise AssertionError
        if not (hasattr(self.clf, '_sigma') and
                self.clf._sigma is not None):
            raise AssertionError
        if not (hasattr(self.clf, 'model_') and
                self.clf.model_ is not None):
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

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        clone_clf = clone(self.clf_ae)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
