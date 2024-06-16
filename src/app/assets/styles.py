def font_size(size="L"):
    if size == "XS":
        return "8px"
    elif size == "S":
        return "10px"
    elif size == "M":
        return "14px"
    elif size == "L":
        return "16px"


def width_px(coeff):
    return f"{1400*coeff}px"


def width_val():
    return 1400
