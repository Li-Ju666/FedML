import matplotlib.pyplot as plt
names = [2, 5, 8, 10, 15, 20, 25, 30, 40, 50]
for i in names:

    text_file = open("./C_implementation/Matrix-based_implementation/error_"+str(i), "r")
    lines = text_file.read().split(' ')
    lines.pop()
    print(len(lines))
    text_file.close()
    data = [float(i) for i in lines]

    plt.plot(data)
    plt.ylabel('Error')
    ##plt.show()
    plt.title("Error ~ iteration with hidden nodes: "+str(i))
    plt.savefig("./C_implementation/Matrix-based_implementation/image"+str(i))
    plt.close()