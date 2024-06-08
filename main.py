# Welcome message
print("Welcome to Burnout Detection System. To proceed - ")

# Prompt for user inputs and store them in variables
gender = input("Enter your Gender (M/F): ")
company_type = input("Enter type of Company (PBC/SBC): ")
wfh_facility = input("Do you have WFH Facility? (Y/N): ")
designation = input("What is your designation? ")
resource_allocation = int(input("Enter your resource allocation level: "))
workload = int(input("Enter your workload: "))
mental_fatigue = workload/10

if designation.lower() == 'trainee':
    designation_value = 0
elif designation.lower() == 'associate':
    designation_value = 1
elif designation.lower() == 'senior associate':
    designation_value = 2
elif designation.lower() == 'manager':
    designation_value = 3
elif designation.lower() == 'director':
    designation_value = 4
elif designation.lower() == 'partner':
    designation_value = 5
else:
    designation_value = 0
designation = designation_value

if gender == "M":
    gender = 1
else:
    gender = 0

if company_type == "PBC": company_type=0
else: company_type=1

if wfh_facility=="Y": wfh_facility=1
else: wfh_facility=0

'''print("\nStored Inputs:")
print(f"Gender: {gender}")
print(f"Company Type: {company_type}")
print(f"WFH Facility: {wfh_facility}")
print(f"Designation: {designation}")
print(f"Resource Allocation: {resource_allocation}")
print(f"Workload: {workload}")'''

print("Analysing... please wait!")
ip = [gender, company_type, wfh_facility, designation, resource_allocation, mental_fatigue, mental_fatigue*2]

'''ip1 = [[0,1,0,2,3,3.8,0.16]]
ip2 = [[0,1,0,3,7,6.9,0.52]]
ip3 = [[0,1,1,2,4,4.4,0.33]]
ip4 = [[1, 1, 1, 4, 8, 8.7, 0.68]]
ip5 = [[1, 1, 1, 2, 4, 8.3, 0.63]]
ip6 = [[1, 0, 0, 3, 7, 6.4, 0.55]]
ip7 = [[1, 1, 0, 2, 6, 5.7, 0.52]]
ip8 = [[1, 0, 1, 2, 5, 7.3, 0.74]]
ip9 = [[1, 1, 1, 2, 5, 5.9, 0.55]]
ip10 = [[1, 1, 0, 1, 2, 4.2, 0.24]]

ip = ip1 + ip2 + ip3 + ip4 + ip5 + ip6 + ip7 + ip8 + ip9 + ip10'''


import pickle
# Specify the filename of the saved model
filename = r'C:\Users\ADMIN\FLPrrojectS\final_saved_model.pkl'

# Open the file in binary read mode and use pickle to deserialize the model
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

'''print(f"Model has been loaded from {filename}")
print(f"Loaded Model: {loaded_model}")'''

print(loaded_model.predict([ip]))

print("Your burnout status is: Burnt Out")



