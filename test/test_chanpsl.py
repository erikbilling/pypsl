
import numpy as np
import unittest
from inputdata import singen
from chanpsl import ChanPsl

class TestChanPsl(unittest.TestCase):

    def setUp(self):
        self.psl = ChanPsl()

    def test_sinus_prediction(self):
        history = []
        N = 10
        for v in singen(stop=N):
            if history:
                self.psl.trainSample(history,target=v)
            history.append(v)
        self.assertEqual(np.round(self.psl.memory.sum()),N*20)

if __name__ == '__main__':
    unittest.main()

