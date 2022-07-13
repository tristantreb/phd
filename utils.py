def getMurdererName(val):
    valid = {0,1}
    if val not in valid:
        raise ValueError("results: status must be one of %r." % valid)
    # use dictionnary mapping
    switcher = {
        0: "Grey",
        1: "Auburn",
    }
    return switcher.get(val)