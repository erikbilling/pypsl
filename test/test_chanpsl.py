
import numpy as np
import unittest
from test.inputdata import singen, trigen
import chanpy
from pypsl.chanpsl import ChanPsl, combine
from pypsl import Library

class TestPsl(unittest.TestCase):
    def setUp(self):
        pass

    def test_nonzero(self):
        a = np.array([[0,1,0],[1,0,1]])
        print(type(np.nonzero(a)))

    def test_encode(self):
        basis = chanpy.Cos2ChannelBasis()
        basis.setParameters(11,0.,1.)
        cv = basis.encode(np.array([[0,1,0],[1,0,1]]))
        print(cv.shape)

    def test_combo(self):
        vlist = [np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9])]
        i = iter(combine(vlist))
        self.assertSequenceEqual(next(i),(1,4,7))
        self.assertSequenceEqual(next(i),(1,4,8))
        self.assertSequenceEqual(next(i),(1,4,9))
        self.assertSequenceEqual(next(i),(1,5,7))
        self.assertSequenceEqual(next(i),(1,5,8))
        self.assertSequenceEqual(next(i),(1,5,9))

    def test_match(self):
        library = Library({(1,): 1, (1,2): 2, (3,): 3})
        psl = ChanPsl(minValue=0, maxValue=10, library=library)
        match = psl.match(psl.encode([1,2,3]))
        self.assertEqual(next(match).rhs,3)
        self.assertEqual(next(match).rhs,2)

    def test_add(self):
        psl = ChanPsl(minValue=0, maxValue=10)
        rhs = psl.__basis__.encode(2)
        lhs = psl.__basis__.encode(3)
        psl.add(rhs,lhs)
        print(psl.library)

    def test_predict(self):
        psl = ChanPsl(minValue=0, maxValue=10)
        rhs = psl.encode(2.)
        lhs = psl.encode(3.)
        psl.add(rhs,lhs)
        self.assertAlmostEqual(psl.predict(rhs),3.)
        
        rhs = psl.encode(5.)
        lhs = psl.encode(7.)
        psl.add(rhs,lhs)
        self.assertEquals(psl.predict(rhs),7.)
        