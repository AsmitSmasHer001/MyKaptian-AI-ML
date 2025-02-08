n = int(input("enter the no of terms "))
a, b = 0, 1
print("Fibonacci sequence up to :",n)
for i in range(n):
    print(a, end=' ')
    a = b 
    b = a + b 
