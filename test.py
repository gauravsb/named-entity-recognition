import re

def remove_puncuation(a):
    res = a
    for i in ",.:!$%#@-":
        res = res.replace(i, "")
    return res


def remove_special_characters(input):
    result = input
    for i in ",.:!$%#@-+":
        result = result.replace(i, "")
    return result

def is_special_character(input):
    if len(input) != 1:
        return False
    res = re.search("[,.:-]+", input)
    return bool(res)

def is_bracket(input):
    if len(input) != 1:
        return False
    return input == "("

if __name__ == "__main__":
    '''
    print("CRICKET".isupper())
    print("cRICKET"[0].isupper())
    print("cRICKET".isupper())
    print("Cricket"[0].isupper())
    print(remove_special_characters("1992").isnumeric())
    print(remove_special_characters("1992:30"))
    print(remove_special_characters("1992:30").isnumeric())
    print(remove_special_characters("1992-03-30"))
    print(remove_special_characters("1992-03-30").isnumeric())
    print(remove_special_characters("12.435353-2525345-345435366-36363"))
    print(remove_special_characters("12.435353-2525345-345435366-36363").isnumeric())
    print(remove_special_characters("1th"))
    print(remove_special_characters("1th").isnumeric())
    '''
    print(is_special_character("-"))
    #print(train)
