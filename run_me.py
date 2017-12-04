__author__ = "Siddharth Chandrasekaran"
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "schandraseka@umass.edu"

import numpy as np
import kaggle
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from itertools import product
import time

# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')

	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()


#Parse param grid
def parse_param_grid(param_grid):
    for p in param_grid:
            items = sorted(p.items())
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params

#Only prints out the top n results based on out of sample error
def get_nbest_result(results,n):
    results.sort(key=lambda tup: tup[0])
    return results[0:n]

#Grid Search CV custom
def cross_validation(model, cv, X, y, paramgridIterator):
    result = []
    for param in paramgridIterator:
        start_time = time.time()
        print("Starting : "+ str(param))
        validation_folds_score = []
        kf = KFold(n_splits=cv, random_state=1, shuffle=True)
        for train_index, test_index in kf.split(X):
            model = DecisionTreeRegressor().set_params(**param)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fittedmodel = model.fit(X_train, y_train)
            ycap = fittedmodel.predict(X_test)
            validation_folds_score.append(compute_error(ycap, y_test))
        result.append((sum(validation_folds_score)/float(len(validation_folds_score)), param))
        print("--- %s seconds ---" % (time.time() - start_time))
    return result                


#Decision Tree                
train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', train_y.shape)




param_grid = {"criterion": ["mae"],
              "splitter": ["best"],
		      "max_depth": [20,25,30,35,40],	
		"min_samples_leaf" :[2],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			
		      }
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
dt = DecisionTreeRegressor()
nbesttreemodel = get_nbest_result(cross_validation(dt, 5, train_x, train_y, paramgridIterator), 10)

for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = dt.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/IndoorLocalization/FinalRes'+str(i)+'.csv';

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)

#Decision Tree
train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('Test=', train_y.shape)



param_grid = {"criterion": ["mae"],
              "splitter": ["best"],
		      "max_depth": [3, 6, 9, 12, 15],	
		"min_samples_leaf" :[2],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			
		      }
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
dt = DecisionTreeRegressor()
nbesttreemodel = get_nbest_result(cross_validation(dt, 5, train_x, train_y, paramgridIterator), 10)

for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = dt.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/PowerOutput/FinalRes'+str(i)+'.csv';

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)

#Knn 
train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', train_y.shape)


param_grid = {"n_neighbors" : [3, 5, 10, 20, 25],
	 "weights" :["distance"],
	"algorithm": ["auto"],    
	"p": [1],
	"metric" : ["manhattan"]     																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																													
}
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
kn = KNeighborsRegressor()
nbesttreemodel = get_nbest_result(cross_validation(kn, 5, train_x, train_y, paramgridIterator), 10)

for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = kn.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/IndoorLocalization/FinalRun'+str(i)+'.csv';

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)
#Knn 
train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('Test=', train_y.shape)

 
param_grid = {"n_neighbors" : [3, 5, 10, 20, 25],
	 "weights" :["distance"],
	"algorithm": ["auto"],    
	"p": [1],
	"metric" : ["manhattan"]     																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																													
}
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
kn = KNeighborsRegressor()
nbesttreemodel = get_nbest_result(cross_validation(kn, 5, train_x, train_y, paramgridIterator), 10)

for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = kn.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/PowerOutput/FinalRun'+str(i)+'.csv';

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)


train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', train_y.shape)

#Lasso regression
param_grid = {"alpha" : [0.0001 , 0.01 , 1, 10],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										
}
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
ls = linear_model.Lasso()
nbesttreemodel = get_nbest_result(cross_validation(ls, 5, train_x, train_y, paramgridIterator), 10)

for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = ls.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/IndoorLocalization/FinalLass'+str(i)+'.csv'

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)

#Ridge regression
param_grid = {"alpha" : [0.0001 , 0.01 , 1, 10]																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										
}
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
ls = linear_model.Ridge()
nbesttreemodel = get_nbest_result(cross_validation(ls, 5, train_x, train_y, paramgridIterator), 10)


for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = ls.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/IndoorLocalization/FinalRidge'+str(i)+'.csv';

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)


train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('Test=', train_y.shape)

#Lasso regression
param_grid = {"alpha" : [0.000001 , 0.0001 , 0.01 , 1, 10],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										
}
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
ls = linear_model.Lasso()
nbesttreemodel = get_nbest_result(cross_validation(ls, 5, train_x, train_y, paramgridIterator), 10)

for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = ls.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/PowerOutput/FinalLass'+str(i)+'.csv';

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)

#Ridge Regression
param_grid = {"alpha" : [0.000001 , 0.0001 , 0.01 , 1, 10],																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										
}
param_grid = dict(param_grid)
paramgridIterator = parse_param_grid([param_grid])
ls = linear_model.Ridge()
nbesttreemodel = get_nbest_result(cross_validation(ls, 5, train_x, train_y, paramgridIterator), 10)

for i,nbesttree in enumerate(nbesttreemodel):
	fittedmodel = ls.set_params(**nbesttree[1]).fit(train_x, train_y)
	print(nbesttree[1])
	print(nbesttree[0])
	test_y =  fittedmodel.predict(test_x)
	file_name = '../Predictions/PowerOutput/FinalRidge'+str(i)+'.csv';

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(test_y, file_name)

