
import numpy as np
import unittest
from inputdata import singen
from pcc import Pcc

class TestChanPsl(unittest.TestCase):

    def setUp(self):
        self.pcc = Pcc()

    def test_sinus_prediction(self):
        history = []
        N = 100
        for v in singen(stop=N):
            if history:
                self.pcc.trainSample(history,target=v)
            history.append(v)
        self.assertEqual(np.round(self.pcc.memory.sum()),N*20)

        data = [v for v in singen(stop=N)]
        result = [self.pcc.predictSample(data[i-1] if i > 0 else None) for i,v in enumerate(data)]
        mse = np.array([(p-v)*(p-v) for p,v in zip(result,data)]).mean()
        self.assertLess(mse,0.000)       

if __name__ == '__main__':
    unittest.main()

