
def lrfill(s,n):
    """
    Returns a string of length n that contains s and whitespaces.

    :param s: string which will be the start of the return string (filled with one whitespace in front)
    :param n: final length of the return string
    :rtype: string
    """
    if len(s) == n:
        return s
    else:
        return " " + s + (n-len(s) - 1)*" "
    
def hLine(m,n):
    """
    Provides a horizontal line of intermitted + and - symbols

    :param m: final length of the ascii line minus 2
    :param n: length of the '-' pieces
    :rtype: string
    """
    return "+" + "+".join(m*[n*"-"]) + "+"

