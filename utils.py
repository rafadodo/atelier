def printMatrix(a):
    for row in a:
        for col in row:
            print("{:.2E}".format(col), end=" ")
        print("")
