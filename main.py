ds = open(r"C:\Users\ADMIN\FLPrrojectS\employee_burnout_analysis.csv")

for line in ds.readlines()[1:]:
    a, b = line.strip().split(",")
    if float(a) < 0.5 and int(b) != 0:
        print("Exception")
        input()
    elif float(a) > 0.5 and int(b) != 1:
        print("Exception")
        input()
    else:
        print(a,b)

