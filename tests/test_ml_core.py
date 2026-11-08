"""Reproducibility and leakage controls. Run: python -m unittest discover -s tests"""
import unittest
import numpy as np
from src.ml_core import (
    read_model, time_splits, prepare_features, with_intercept,
    objective_and_gradient, gd_step, train_gd, probabilities,
    DEFAULT_FEATURES, TARGET,
)

class CreditProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits=time_splits(read_model())

    def test_temporal_labels_available_before_next_period(self):
        tr=self.splits['train'];va=self.splits['validation'];te=self.splits['test']
        self.assertTrue((tr.LabelReadyDate <= '2023-01-01').all())
        self.assertTrue((va.LabelReadyDate <= '2025-01-01').all())
        self.assertTrue((te.SnapshotDate >= '2025-01-01').all())
        self.assertTrue((te.IsMatureTarget == 1).all())
        self.assertTrue(set(tr.CreditID).isdisjoint(set(va.CreditID)))
        self.assertTrue(set(tr.CreditID).isdisjoint(set(te.CreditID)))
        self.assertTrue(set(va.CreditID).isdisjoint(set(te.CreditID)))

    def test_preprocessing_and_features(self):
        f=prepare_features(self.splits,DEFAULT_FEATURES)
        self.assertEqual(len(f.x_train),len(self.splits['train']))
        self.assertTrue(np.isfinite(f.x_train).all())
        self.assertTrue(np.isfinite(f.x_valid).all())
        with self.assertRaises(ValueError):
            prepare_features(self.splits,['CustomerID',TARGET])

    def test_gradient_matches_finite_difference(self):
        X=np.array([[1,0.3,-0.2],[1,-1.2,2.4],[1,0.2,0.8],[1,1.,1.]])
        y=np.array([0,1,1,0],dtype=float)
        theta=np.array([-.2,.3,-.7],dtype=float)
        loss,gradient=objective_and_gradient(X,y,theta,l2=2)
        epsilon=1e-6
        for i in range(len(theta)):
            up=theta.copy();down=theta.copy();up[i]+=epsilon;down[i]-=epsilon
            numerical=(objective_and_gradient(X,y,up,l2=2)[0]-
                       objective_and_gradient(X,y,down,l2=2)[0])/(2*epsilon)
            self.assertAlmostEqual(gradient[i],numerical,places=6)

    def test_gd_step_reduces_train_loss(self):
        f=prepare_features(self.splits,['MaxHistoricalDPD','DebtToIncomeRatio'])
        X=with_intercept(f.x_train)
        t=np.zeros(3)
        start,_=objective_and_gradient(X,f.y_train,t)
        for _ in range(15):
            t,_,_=gd_step(X,f.y_train,t,learning_rate=.2)
        end,_=objective_and_gradient(X,f.y_train,t)
        self.assertLess(end,start)

    def test_training_yields_epochs_and_weights(self):
        f=prepare_features(self.splits,DEFAULT_FEATURES[:2])
        results=list(train_gd(f,lr=.1,epochs=5,checkpoint_every=1))
        self.assertEqual(results[-1][0],5)
        theta=results[-1][1]
        self.assertEqual(len(theta),3)
        p=probabilities(f.x_test,theta)
        self.assertTrue(((p>=0)&(p<=1)).all())

if __name__=='__main__':
    unittest.main()
