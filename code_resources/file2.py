import sys

torte = ['tiramisu', 'torta di mele', 'marmorizzata']
ingredienti = ['farina', 'mele', 'cacao']

def zipme():
    '''zip test'''
    for el in zip(torte, ingredienti):
        print (el)
        print(type(el))

def squaregen(start, stop):
    '''Generator test'''
    for i in range(start, stop):
        print("generating square of {}".format(i))
        yield(i*i)

class Square:
    '''Iterator test'''
    def __init__(self, start, stop):
        self._start = start
        self._stop = stop

    def __iter__(self):
        return self

    def __next__(self):  # Python 2: def next(self)
        if self._start >= self._stop:
            raise StopIteration
        current = self._start * self._start
        self._start += 1
        return current


class Bar(object):
    '''Iterator test'''
    def __init__(self):
       self.idx = 0
       self.data = range(4)

    def __iter__(self):
       print("__iter__ called")
       return self

    def __next__(self):
       print("__next__ called")
       self.idx += 1
       try:
           return self.data[self.idx-1]
       except IndexError:
           self.idx = 0
           raise StopIteration  # Done iterating.
    next = __next__  # python2.x compatibility.


def main(arg):
    print("Zip test")
    zipme()

    print("Generatos vs Iterators")
    # generator
    for i in squaregen(5,10):
        print(i)

    # inline generator
    for i in(k*k for k in range(5, 10)):
        print("inline generator gives {}".format(i))

    aq_range = Square(5,10)
    for i in aq_range:
        print ("Usinc class iterator object {}".format(i))

    bb = Bar()
    print(list(bb))

    return 0


if __name__ == "__main__":
    print("Called as main!")
    main(sys.argv)