import collections
import sys

Card = collections.namedtuple('SingleCard', ['rank','suit'])

class FrenchDeck:
    ranks = [ str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for rank in self.ranks
                                        for suit in self.suits]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]


def main(arg):
    # single card
    beer_card = Card('7', 'diamonds')
    print(beer_card)

    # deck of cards
    deck = FrenchDeck()
    print("deck len is:{} ".format(len(deck)))

    #ops on the deck so defined: note that are supported because of the implemented method __len__ and __getitem__
    # Traceback(most; recent; call; last): File; "/Users/ANDREA/PycharmProjects/LearningPython3/file1.py", line; 44,
    # in < module >; main(sys.argv)
    # File; "/Users/ANDREA/PycharmProjects/LearningPython3/file1.py", line; 33, in main
    # print("random: {}".format(choice(deck)))
    # File; "/Users/ANDREA/anaconda/lib/python3.6/random.py", line; 258, in choice
    # return seq[i]
    # TypeError: 'FrenchDeck'; object; does; not support; indexing



    from random import choice
    print("random: {}".format(choice(deck)))

    print("direct get specific item: {}".format(deck[0]))
    return 0





if __name__ == "__main__":
    print("Called as main!")
    main(sys.argv)