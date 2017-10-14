''' PES College of Engineering
    8th Semester Project
    Cholera prediction
''' 
# Check the versions of libraries 
# Python version

from multiprocessing import Process

def print_platformversions():

	print('Python: {}'.format(sys.version))
	print("\n")
	print('scipy: {}'.format(scipy.__version__))	
	print("\n")
	print('numpy: {}'.format(numpy.__version__))
	print("\n")
	print('matplotlib: {}'.format(matplotlib.__version__))
	print("\n")
	print('pandas: {}'.format(pandas.__version__))
	print("\n")
	print('sklearn: {}'.format(sklearn.__version__))
	print("\n")
	return

#Read file 

''' load dataset
	sex -
    0 - male
    1 - female
'''

def readfile_csv(filename):

	file = filename
	names = ['age', 'sex', 'vomiting', 'diarrohea', 'result']
	dataset = pandas.read_csv(file, names=names)
	return dataset


def newline():
	print("\n")


# Print shape,tuple and descriptions
def printshape(dataset):

	print ("(instances, attributes )",  dataset.shape)
	print("\n")
	print(dataset.head(30));
	print("\n")
	print(dataset.describe())
	print("\n")
	return dataset

#Clas distribution
def display_data(dataset):

	print(dataset.groupby('age').size())
	print("\n")
	print(dataset.groupby('sex').size())
	print("\n")
	print(dataset.groupby('result').size())
	return dataset

# Plot histogram graph
'''def histogam(dataset):
	dataset.hist()
	plt.show()

def box(dataset):
	dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
	plt.show()
'''

# scatter plot matrix
def scattergr(dataset):

	scatter_matrix(dataset)
	plt.show()


#function to check correctness of user inputs
def chk_input(usr_name, usr_age, usr_sex, usr_vomit, usr_diarr):

    if(type(usr_name) != str):
        print("Invalid type! Please enter string")
        return False
    elif(type(usr_age) != int):
        print("Invalid age type! Please enter valid integer")
        return False
    elif(type(usr_vomit) != int):
        print("Invalid type! Please enter valid integer(vomit in no. of hours)")
        return False
    elif(type(usr_diarr) != int):
        print("Invalid type! Please enter valid integer(diarrohea in no. of hours)")
        return False
    
    newline()
    return True


#test module
def test_module(cart):
	test_data = [[23, 0, 6, 5], [33, 0, 15, 11]]
	test_prediction = cart.predict(test_data)
	if(test_prediction[0] != 'Negative' or test_prediction[1] != 'Positive'):
		print("Test failed!")
	else:
		dataset = readfile_csv("mydata.csv")
		# Use Print shape ,Tuple
		dataset = printshape(dataset)
		# Use Class distributon
		datset = display_data(dataset)
		print("Test passed!")
	newline()


#user module
def usr_module(cart):

    usr_name = input("Enter patient's name - ")
    newline()
    
    usr_age = int(input("Enter patient's age - "))
    newline()
    
    usr_sex = input("Enter patient's sex - ")
    newline()

    if(usr_sex == "FEMALE" or usr_sex == "female" or usr_sex == 'F' or usr_sex == 'f'):
        usr_sex = 0
    elif(usr_sex == "MALE" or usr_sex == "male" or usr_sex == 'M' or usr_sex == 'm'):
        usr_sex = 1
    else:
        sys.exit("Please enter valid gender - M, m, male, MALE, F, f, female, FEMALE")
    newline() 

    usr_vomit = int(input("Enter no. of hours of vomit - "))
    newline()    

    usr_diarr = int(input("Enter no. of hours of diarrohea - "))
    newline()
    
    if(chk_input(usr_name, usr_age, usr_sex, usr_vomit, usr_diarr) == False):
        print("Error in user information")
        return
    else:
        usr_data = []
        usr_tmp = []
        usr_tmp.append(usr_age)
        usr_tmp.append(usr_sex)
        usr_tmp.append(usr_vomit)
        usr_tmp.append(usr_diarr)
        usr_data.append(usr_tmp)
        # usr_data = [[23, 0, 6, 5]]
        usr_prediction = cart.predict(usr_data)
        print("Your result is " + usr_prediction[0])
        newline()
        for usr_result in usr_prediction:
            if usr_result == "Positive":
                print("Dear " + usr_name + ", Please see the doctor soon")
            else:
                print("Dear " + usr_name + ", No worries :)")
        newline()


# Split-out validation dataset
def split_vaidation_dataset(dataset):

	array = dataset.values
	# print(array)

	X = array[:,0:4]
	Y = array[:,4]
	validation_size = 0.20
	seed = 7
	scoring = 'accuracy'
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

	models = []

	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	results = []
	names = []

	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()
	cart = DecisionTreeClassifier()
	cart.fit(X_train, Y_train)
	predictions = cart.predict(X_validation)

	# print(predictions)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
	newline()

	if(len(sys.argv) > 1 and sys.argv[1] == '-t'):
	    test_module(cart)
	else:
	    usr_module(cart)



if __name__ == "__main__":

	import sys,scipy,numpy,matplotlib,pandas,sklearn
	from pandas.tools.plotting import scatter_matrix
	import matplotlib.pyplot as plt
	from sklearn import model_selection
	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import accuracy_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.naive_bayes import GaussianNB
	from sklearn.svm import SVC

	print_platformversions()
	# Read a file rom CMD
	# arg = sys.argv[0:]
	dataset = readfile_csv("mydata.csv")
	# Plot histogram
	# histogam(dataset)
	# Plot box
	# box(dataset)
	# Plot scatter
	scattergr(dataset)
	# Split-out validation dataset
	split_vaidation_dataset(dataset)
