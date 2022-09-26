import numpy as np


class transform_set_val():
    def __init__(self, base_learner):
        self.base_learner = base_learner

    def fit(self, X_unlab, sigma=0):
        self.sigma = sigma
        self.p_matr = self.base_learner.predict_proba(X_unlab)
        n, K = self.p_matr.shape
        self.p_matr += np.random.normal(0, self.sigma, (n, K))

    def predict(self, X_test, s=1):
        X_proba = self.base_learner.predict_proba(X_test)
        n, K = X_proba.shape
        X_proba += np.random.normal(0, self.sigma, (n, K))
        N, _ = self.p_matr.shape
        self.decision = np.sum((X_proba[:,:,None,None] < self.p_matr), axis=(2,3)) / N
        return (self.decision <= s)

    def predict_new_s(self, X_test, s):
        return (self.decision <= s)
  

class transform_set_val_conformal():
    def __init__(self, base_learner):
        self.base_learner = base_learner

    def fit(self, X_unlab, sigma=0):
        self.sigma = sigma
        self.p_matr = self.base_learner.predict_proba(X_unlab)
        n, K = self.p_matr.shape
        self.p_matr += np.random.normal(0, self.sigma, (n, K))
        self.U = np.random.choice(range(K), n)

    def predict(self, X_test, s=1):
        X_proba = self.base_learner.predict_proba(X_test)
        n, K = X_proba.shape
        alpha = s / K
        X_proba += np.random.normal(0, self.sigma, (n, K))
        N, _ = self.p_matr.shape
        self.decision = np.sum((X_proba[:,:,None,None] < self.p_matr), axis=(2,3)) / N
        return (self.decision <= s)

    def predict_new_s(self, X_test, s):
        return (self.decision <= s)