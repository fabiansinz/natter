
def lrfill(s,n):
    if len(s) == n:
        return s
    else:
        return " " + s + (n-len(s) - 1)*" "
    
def hLine(m,n):
    return "+" + "+".join(m*[n*"-"]) + "+"

