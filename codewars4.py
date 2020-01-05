def solution(s):                                                        #This is the translation function
    original_letters='abcdefghijklmnopqrstuvwxyz'                       #The original letters to be decoded, the lowercase alphabet
    decoded_letters=original_letters[::-1]                              #The letters we will change them to, the reverse lowercase alphabet
    translation_key=str.maketrans(original_letters, decoded_letters)    #Create a translation key
    decoded_message=s.translate(translation_key)                        #Translate the coded_message using the translation key
    print(decoded_message)
    print(decoded_message=='did you see last night\'s episode?')
    return decoded_message                                              #Return the decoded message

solution("wrw blf hvv ozhg mrtsg'h vkrhlwv?")
solution("vmxibkgrlm")