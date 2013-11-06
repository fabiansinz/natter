
def lrfill(s,n):
    """
    Returns a string of length n that contains s and whitespaces.

    :param s: string which will be the start of the return string (filled with one whitespace in front)
    :type s: string
    :param n: final length of the return string
    :type n: int
    :returns: whitespace padded string
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
    :type m: int
    :param n: length of the '-' pieces
    :type n: int
    :returns: string with intermitted + and - chars
    :rtype: string
    """
    return "+" + "+".join(m*[n*"-"]) + "+"

