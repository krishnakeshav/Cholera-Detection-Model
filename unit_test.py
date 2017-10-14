''' Unit test for correctness of data in CSV file
    PES College of Engineering
    8th Semester Project
    Cholera prediction
''' 

import pandas
import csv
import sys

def readfile_csv(filename):

	file = filename
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		dataset = list(reader)

	line = 0
	flag = True
	col = 0
	print(dataset)

	for tuples in dataset:
		line = line + 1
		if(tuples[0] <= '0'):
			flag = False
			col = 1
		elif(tuples[1] != '0' and tuples[1] != '1'):
			flag = False
			col = 2
		elif(tuples[2] <= '-1'):
			flag = False
			col = 3
		elif(tuples[3] <= '-1'):
			flag = False
			col = 4
		elif(tuples[4] != 'Positive' and tuples[4] != 'Negative'):
			flag = False
			col = 5

	if(flag == False):
		print("Invalid instance in line number ", line)
		print("Column ",)
		print(col)
		sys.exit("Please check CSV file again!")
		return
	else:
		print("Tests succesfully passed!")


if __name__ == "__main__":
	readfile_csv("mydata.csv")
