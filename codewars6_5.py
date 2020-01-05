import itertools

def perm(string):


    print(set(''.join(x) for x in itertools.permutations(string)))

perm('aabb')