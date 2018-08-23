
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