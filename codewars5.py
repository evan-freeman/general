import string

def find_missing_letter(chars):
    for i in range(len(chars)-1):
        next_letter=string.ascii_letters[string.ascii_letters.find(chars[i+1])]
        if chars[i+1]!=next_letter:
            return next_letter

find_missing_letter(['a','b','c','d','f'])