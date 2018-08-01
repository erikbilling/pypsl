
import numpy as np
import unittest
from test.inputdata import singen
from pcc import Pcc

class TestPcc(unittest.TestCase):

    def setUp(self):
        pass

    def test_zero_prediction(self):
        pcc = Pcc(11,-1,1)
        self.assertEqual(pcc.predict(0),0.0)

    def test_encode(self):
        N = 11
        pcc = Pcc(N,-1,1)
        p = pcc.encode(0)
        self.assertEqual(p.sum(),1.5)
        self.assertEqual(p.size,N)

    def test_encode_list(self):
        N = 11
        pcc = Pcc(N,-1,1)
        p = pcc.encode([0])
        self.assertEqual((p-pcc.encode(0)).sum(),0)

    def test_training_error(self):
        pcc = Pcc(25,-1,1)
        N = 1000
        data = [v for v in singen(length=N)]
        for trace,v in pcc.trace(data):
            pcc.train(trace,v)

        result = [pcc.predict(trace) for trace,v in pcc.trace(data)]
        mse = np.square(np.array(result)-np.array(data)).mean()
        print('MSE: {0:.6f}'.format(mse))
        self.assertLess(mse,0.001)

    def test_noise_error(self):
        pcc = Pcc(25,-1,1,noise=0.1)
        N = 1000
        data = [v for v in singen(length=N)]
        for trace,v in pcc.trace(data):
            pcc.train(trace,v)

        result = [pcc.predict(trace) for trace,v in pcc.trace(data)]
        mse = np.square(np.array(result)-np.array(data)).mean()
        print('MSE: {0:.6f}'.format(mse))
        self.assertLess(mse,0.005)

    def test_dot(self):
        N = 11
        pcc = Pcc(N,-1,1)
        v = pcc.encode(0.1)
        pv = np.dot(v,pcc.memory)

    def test_trace(self):
        # p2 = Pcc(25,-1,1)
        # trace = InputTrace(p2)
        # trace.train(data)     
        pass   

if __name__ == '__main__':
    unittest.main()

