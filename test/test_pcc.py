
import numpy as np
import unittest
from test.inputdata import singen
from pcc import Pcc

class TestChanPsl(unittest.TestCase):

    def setUp(self):
        self.pcc = Pcc(25,-1,1)

    def test_sinus_prediction(self):
        history = []
        N = 100
        data = [v for v in singen(stop=N)]
        for v in data:
            if history:
                self.pcc.trainSample(history,target=v)
            history.append(v)
        self.assertEqual(np.round(self.pcc.memory.sum()),N*20)
        result = [self.pcc.predictSample(data[i-1] if i > 0 else None) for i,v in enumerate(data)]
        mse = np.array([(p-v)*(p-v) for p,v in zip(result,data)]).mean()
        self.assertLess(mse,0.01)       
        

if __name__ == '__main__':
    unittest.main()

