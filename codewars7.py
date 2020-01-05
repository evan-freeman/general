import math

def sum_for_list(lst):
    #We only need to check up to one third of the biggest number in the list
    #i.e. the worst case scenario is that num=3* some prime. if it was any more, it would be even
    biggest=max(abs(x) for x in lst)
    primes=[]
    to_check=range(2, math.ceil(biggest/3)+1)

    #Here we generate out list of possible prime factors
    for x in to_check:
        x_is_prime=True
        x_to_check=range(2, math.ceil(x/3)+1)
        for y in x_to_check:
            if x%y==0:
                x_is_prime=False
        if x_is_prime:
            primes.append(x)

    #Here we make our list
    solution=[]
    for x in primes:
        mult_of_x=[]    #Here we'll store all multiples of x from out list
        for y in lst:
            if y%x==0:
                mult_of_x.append(y)
        if len(mult_of_x)!=0:
            solution.append([x, sum(mult_of_x)])

    print(solution)
    return(solution)

l=list(range(-999,-22))
print(l)
sum_for_list(l)