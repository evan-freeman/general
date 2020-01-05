import itertools


def next_bigger(n):
    # make all permutations
    l = list(str(n))
    p = [''.join(x) for x in itertools.permutations(l)]
    p_int = [int(x) for x in p]

    # remove dupes
    # set()
    unique = list(set(p_int))

    # order them
    # sorted()
    ordered = sorted(unique)
    print(ordered)

    # index n in ordered
    i = ordered.index(n)

    # find length of ordered
    length = len(ordered)

    # if there's more than 1, return the next indexed one
    if length > 1 and i != length - 1:
        print(ordered[i + 1])
        return ordered[i + 1]

    # otherwise return -1
    else:
        print(-1)
        return -1

next_bigger(1441)
next_bigger(1234555222)

