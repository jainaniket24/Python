names =  input("enter names of students separated by comma: ").title().split(',')
assignments = input("enter number of missing assignments separated by comma: ").split(',')
grades = input("enter grades separated by comma: ").split(',')
new_grades = input("grades after assignment separated by comma: ").split(',')

# message string to be used for each student

message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. You're current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

for i in range(len(names)):
    print(message.format(names[i], assignments[i], grades[i], new_grades[i]))