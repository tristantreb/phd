def weight(weight, id):
    if weight < 30 or weight > 122:
        print(
            "Warning - ID {} has Weight ({}) outside 30-122 kg range".format(id, weight)
        )
    return -1
