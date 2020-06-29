import matplotlib.pyplot as plt
text_file = open("./C_implementation/Matrix-based_implementation/errors.txt", "r")
lines = text_file.read().split(' ')
print(lines)
print(len(lines))
text_file.close()
print(type(lines))
print(type(lines[0]))
data = [float(i) for i in lines[:20000]]
print(type(data))
print(type(data[0]))

plt.plot(data)
plt.ylabel('Error')
##plt.show()
plt.savefig("./C_implementation/Matrix-based_implementation/Error_change.png")