
import numpy as np
import unittest
from test.inputdata import singen, trigen
from pcc import Pcc, UniformChannelEncoder, LogDiffEncoder, MeanBuffer, integrate

class TestPcc(unittest.TestCase):

    def setUp(self):
        pass

    def test_zero_prediction(self):
        pcc = Pcc(11,-1,1)
        self.assertEqual(pcc.predict(0),0.0)

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
        pcc = Pcc(UniformChannelEncoder(25,-1,1,noise=0.05))
        N = 1000
        data = [v for v in singen(length=N)]
        for trace,v in pcc.trace(data):
            pcc.train(trace,v)

        result = [pcc.predict(trace) for trace,v in pcc.trace(data)]
        mse = np.square(np.array(result)-np.array(data)).mean()
        print('MSE: {0:.6f}'.format(mse))
        self.assertLess(mse,0.05)

    def test_sequence_reconstruction(self):
        pcc = Pcc(25,-1,1)
        N = 1000
        data = [v for v in trigen(length=N)]
        for trace,v in pcc.trace(data):
            pcc.train(trace,v)

        result = [v for trace,v in pcc.gen(data)]
        mse = np.square(np.array(result)-np.array(data)).mean()
        print('MSE: {0:.6f}'.format(mse))
        self.assertLess(mse,0.005)


class TestUniformEncoder(unittest.TestCase):

    def test_encode(self):
        N = 11
        encoder = UniformChannelEncoder(N,-1,1)
        p = encoder.encode(0)
        self.assertEqual(p.sum(),1.5)
        self.assertEqual(p.size,N)

    def test_encode_list(self):
        N = 11
        encoder = UniformChannelEncoder(N,-1,1)
        p = encoder.encode([0])
        self.assertEqual((p-encoder.encode(0)).sum(),0)

class TestLogDiffEncoder(unittest.TestCase):

    def test_reshape(self):
        encoder = LogDiffEncoder(20,-1,1)
        for v in np.arange(-1,1,0.1):
            self.assertAlmostEqual(v,encoder.restore(encoder.reshape(v)))

    def test_encode(self):
        encoder = LogDiffEncoder(11,-1,1)
        for v in np.arange(-1,1,0.1):
            self.assertAlmostEqual(v,encoder.decode(encoder.encode(v)))

    def test_diff_learning(self):
        pcc = Pcc(LogDiffEncoder(25,-1,1))
        N = 1000
        data = [v for v in singen(length=N)]
        for trace,v,dv in pcc.trace(data):
            pcc.train(trace,dv)

        dvResult = [pcc.predict(trace) for trace,v,dv in pcc.trace(data)]
        result = np.array(dvResult)+np.array(data)
        mse = np.square(result[:-1]-np.array(data[1:])).mean()
        print('MSE: {0:.6f}'.format(mse))
        self.assertLess(mse,0.01)

class TestIntegrate(unittest.TestCase):

    def test_integration(self):
        for a,b in zip(integrate([1,2,3],0),[1,3,6]):
            self.assertEqual(a,b)

class TestMeanBuffer(unittest.TestCase):

    def test_mean(self):
        b = MeanBuffer(5)
        self.assertEqual(b.mean(),0)
        for i in range(5):
            b.put(i)
        self.assertEqual(b.mean(),2.)

if __name__ == '__main__':
    unittest.main()

