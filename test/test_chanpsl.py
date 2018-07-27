
import unittest
from inputdata import singen
from chanpsl import ChanPsl

class TestChanPsl(unittest.TestCase):

    def setUp(self):
        self.psl = ChanPsl()

    def test_sinus_prediction(self):
        history = []
        for v in singen():
            if history:
                result = self.psl.trainSample(history,target=v)
            history.append(v)


if __name__ == '__main__':
    unittest.main()

