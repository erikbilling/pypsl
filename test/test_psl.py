
import numpy as np
import unittest
from test.inputdata import singen, trigen
from pypsl import Psl, Library

class TestPsl(unittest.TestCase):
    def setUp(self):
        pass

    def test_library(self):
        lib = Library({'A': 'B'})
        print (lib)
        self.assertEquals(lib.get('A'),'B')

    def test_match(self):
        lib = Library({'A': 'B', 'D': 'E', 'CD': 'E', 'BCD': 'E'})
        lib.add('D','X')
        psl = Psl(lib)
        print (list(psl.match('ABCD')))
        self.assertEquals(len(list(psl.match('ABCD'))),4)
        #lib['A'] = 'B'
        #psl = Psl(lib)

    def test_lib_len(self):
        lib = Library([('A','B'), ('A', 'C'), ('B', 'C')])
        self.assertEquals(len(lib),3)        

    def test_select(self):
        lib = Library({'A': 'B', 'D': 'E', 'CD': 'E', 'BCD': 'E'})
        lib.add('D','X',hits=2)
        psl = Psl(lib)
        self.assertEquals(psl.select('D').rhs,'X')

    def test_predict(self):
        lib = Library({'A': 'B', 'D': 'E', 'CD': 'E', 'BCD': 'E'})
        lib.add('D','X',hits=2)
        psl = Psl(lib)
        self.assertEquals(psl.predict('D'),'X')
        self.assertEquals(psl.predict('CD'),'E')

    def test_train(self):
        s = 'abccabccabccabcc'
        psl = Psl()
        psl.train(s)
        print('Sequence "{}" yields library:'.format(s),psl.library)
        self.assertEquals(len(psl.library), 6)

        # for i in range(100): 
        #     c = psl.predict(s)
        #     print(i,'Predicted:',c)
        #     s += c

    def test_number_prediction(self):
        x = list(trigen(length=30,amplitude=3))
        psl = Psl()
        psl.train(x)
        self.assertEquals(psl.predict([0,1,2]),3)
        self.assertEquals(psl.predict([1,0]),-1)
        # print (psl.library)

        # y = x[:2]
        # while len(y) < len(x): 
        #     v = psl.predict(y)
        #     print('Predicted:',v)
        #     y.append(v)
        # print('Training sequence: ',x)
        # print('Generated sequence:',y)