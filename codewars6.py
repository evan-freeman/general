import itertools
import collections

def permutations(string):
    # step 1: make the permutations
    l = list(string)
    p = itertools.permutations(l)
    p_list = [''.join(x) for x in p]

    # step 2: remove duplicates
    count=collections.Counter(p_list)
    no_dupes=[name for name in count]
    print(count)
    print(no_dupes)
    return no_dupes