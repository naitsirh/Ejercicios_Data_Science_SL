#from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.datasets import load_boston
#from sklearn.linear_model import LinearRegression




'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *   D A T A  M A N I P U L A T I O N   *  *  *  *  *  *  
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''


'''
heights = [189, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 
	178, 183, 193, 178, 173, 174, 183, 183, 180, 168, 180, 170, 178, 182, 180, 
	183, 178, 182, 188, 175, 179, 183, 193, 182, 183, 177, 185, 188, 188, 182, 
	185, 191]

heights_arr = np.array(heights)

#heights_arr[3] = 165

#print(heights_arr)
#print(sum(heights)/45)

cnt = 0
for height in heights:
	if height > 188:
		cnt += 1
print(cnt)

print(sum([(x > 188) for x in heights]))




print((heights_arr > 188).sum())
print(heights_arr.mean())
print(heights_arr.std())
print(heights_arr.var())
print(heights_arr)




heights_arr = np.array(heights)
print(heights_arr.size)
print(heights_arr.shape)
print(len(heights_arr))




ages = [57, 61, 57, 57, 58, 57, 61, 54, 68, 51, 49, 64, 50, 48, 65, 52, 56, 46, 
	54, 49, 51, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 51, 60, 62, 43, 55, 56, 
	61, 52, 69, 64, 46, 54, 47, 70]

heights_and_ages = heights + ages

heights_and_ages_arr = np.array(heights_and_ages)

#print(heights_and_ages_arr.shape)
#print(heights_and_ages_arr.reshape(2,45))
#print((heights_and_ages_arr.reshape(2,45)).shape)




ages_arr = np.array(ages)

#print(ages_arr.shape)
#print(ages_arr[:5,])




heights_arr = heights_arr.reshape((1,45))
ages_arr = ages_arr.reshape((1,45))

height_age_arr = np.vstack((heights_arr, ages_arr))

print(height_age_arr.shape)
print(height_age_arr[:, :3])




heights_arr = heights_arr.reshape((45,1))
ages_arr = ages_arr.reshape((45,1))

#height_age_arr = np.hstack((heights_arr, ages_arr))




#height_age_arr = np.concatenate((heights_arr, ages_arr), axis=1)
height_age_arr = np.concatenate((heights_arr, ages_arr), axis=0)

print(height_age_arr)
print()
print(height_age_arr[:,:3])
'''




#print(heights_arr.dtype)

heights_float = [189.0, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 
	175, 178, 183, 193, 178, 173, 174, 183, 183, 180, 168, 180, 170, 178, 182, 
	180, 183, 178, 182, 188, 175, 179, 183, 193, 182, 183, 177, 185, 188, 188, 
	182, 185, 191]

#heights_float_arr = np.array(heights_float)

'''
print(heights_float_arr)
print()
print(heights_float_arr.dtype)
print(heights_float_arr > 188.)
'''




#print(heights_arr[2])

#heights_and_ages_arr = heights_and_ages_arr.reshape(2,45)

#print(heights_and_ages_arr[1,2])
#print(heights_and_ages_arr[0, 0:3])
#print(heights_and_ages_arr[0, :3])
#print(heights_and_ages_arr[:, 3])

#heights_and_ages_arr[0, 3] = 165
#heights_and_ages_arr[0,:] = 180
#heights_and_ages_arr[:2, :2] = 0

#print(heights_and_ages_arr[:, :2])
#print(heights_and_ages_arr[:, :2])




#heights_and_ages_arr[:, 0] = [190, 58]

#new_record = np.array([[111, 222, 333], [11, 22, 33]])
#heights_and_ages_arr[:, 42:] = new_record

#print(heights_and_ages_arr)



'''
heights_arr = np.array([189, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 
	173, 175, 178, 183, 193, 178, 173, 174, 183, 183, 180, 168, 180, 170, 178, 
	182, 180, 183, 178, 182, 188, 175, 179, 183, 193, 182, 183, 177, 185, 188, 
	188, 182, 185, 191])

ages_arr = np.array([57, 61, 57, 57, 58, 57, 61, 54, 68, 51, 49, 64, 50, 48, 
	65, 52, 56, 46, 54, 49, 51, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 51, 
	60, 62, 43, 55, 56, 61, 52, 69, 64, 46, 54, 47, 70]).reshape((-1,1))
'''

#print(heights_arr)
#print(ages_arr)

#heights_arr = heights_arr.reshape((45,1))
#height_age_arr = np.hstack((heights_arr, ages_arr))

#print(heights_arr)
#print(height_age_arr)
#print(height_age_arr[:,0]*0.0328084)
#print(height_age_arr)

#print(height_age_arr.sum(axis=0))




#print(height_age_arr[:, 1] < 55)
#print(height_age_arr[:, 1] == 51)
#print((height_age_arr[:, 1] == 51).sum())




#mask = height_age_arr[:, 0] >= 182
#print(mask.sum())




#tall_presidents = height_age_arr[mask, ]
#print(tall_presidents)
#print(tall_presidents.size)




#mask = (height_age_arr[:, 0] >= 182) & (height_age_arr[:, 1] <= 50)
#print(height_age_arr[mask,])



'''
n, p = [int(x) for x in input().split()]

listx = []

for i in range(n):
	listx += [float(k) for k in input().split()]

array = np.array(listx).reshape((n, p))

print(array.mean(axis=1).round(2))




listx = []

for i in range(n):
	listx.append(input().split())

print(np.array(listx).astype(np.float16).mean(axis=1).round(2))
'''



'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  *    D A T A   A N A L I S Y S   *  *  *  *  *  *  *
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''


'''
print(pd.Series([1, 2, 3], index=['a', 'b', 'c']))
print()
print(pd.Series(np.array([1, 2, 3]), index=['a', 'b', 'c']))
print()
#print(pd.Series({'a': 1, 'b': 2, 'c': 3}))

series = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(series['a'])

a = pd.Series([1,2,3])
b = pd.Series([4,5,6])

print(a+b)




wine_dict = {
	'red_wine': [3, 6, 5],
	'white_wine': [5, 0, 10]
}

sales = pd.DataFrame(wine_dict, index=['hans', 'bob', 'charles'])
'''

#print(sales)
#print(sales['white_wine'])




#presidents_df = pd.read_csv(
#	'https://sololearn.com/uploads/files/president_heights_party.csv’, index_col='name')

#presidents_df = pd.read_csv('president_heights_party.csv', index_col='name')

#print(presidents_df.size)
#print(presidents_df.shape)
#print(presidents_df.shape[1])




#print(presidents_df.head(n=3))
#print(presidents_df.tail())

#presidents_df.info()




#print(presidents_df.loc['Abraham Lincoln'])
#print(type(presidents_df.loc['Abraham Lincoln']))
#print()
#print(presidents_df.loc['Abraham Lincoln'].shape)
#print(presidents_df.loc['Abraham Lincoln':'Ulysses S. Grant'])




#print(presidents_df.iloc[15])
#print(presidents_df.iloc[15:18])




#print(presidents_df.columns)
#print(presidents_df['height'])
#print()
#print(presidents_df['height'].shape)
#print(presidents_df['height'].size)

#print(presidents_df[['height', 'age']].head(n=3))




#print(presidents_df.loc[:, 'order':'height'].head(n=3))




#print(presidents_df.min())
#print(presidents_df.max())
#print(presidents_df['age'].mean())
#print(presidents_df.var())
#print(presidents_df.std())
#print(presidents_df.median())




#series = pd.Series([1, 2, 1, 4, 2])
#series = pd.Series([1, 2, 4, 2, 4, 1])
#print(series.median())
#print(len(series))
#print(series.quantile([0.5]))




#print(presidents_df['age'].quantile([0.25, 0.5, 0.75, 1]))
#print(presidents_df['age'].quantile(0.5))
#print(presidents_df['age'].mean())
#print()
#print(presidents_df['age'].median())



#const = pd.Series([2, 2, 2])
#print('Median is: ' + str(const.median()))
#print(const.var())
#print(const.std())




#dat = pd.Series([2, 3, 4])
#print(dat.mean())
#print(dat.var())
#print(dat.std())

#print(presidents_df['age'].var())
#print(presidents_df['age'].std())
#print(presidents_df.std())




#print(presidents_df['age'].describe())
#print(presidents_df.describe())




#print(presidents_df['party'].value_counts())
#print(presidents_df['party'].describe())




#print(presidents_df.groupby('party')['age'].mean())
#print(presidents_df.groupby('party').mean())

#print(presidents_df.groupby('party')['height'].agg(['min', 'median', 'max']))
#print(presidents_df.groupby('party')['age'].agg(['min', 'median', 'max']))

#print(presidents_df.groupby('party').agg({
#		'height': ['median', 'mean'],
#		'age': [min, max]
#		}))



'''
r = int(input()) 
lst = [float(x) for x in input().split()]
arr = np.array(lst)

arr = arr.reshape(r, int(len(lst)/r))

print(arr.round(2))




r = 2
lst = [3.321, 4.432, 5.543, 6.656, 7.765, 8.876]
arr = np.array(lst)

arr = arr.reshape(r, int(len(lst)/r))

print(arr.round(2))
'''


'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *   D A T A   V I S U A L I Z A T I O N   *  *  *  *  *  *
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''

plt.style.use('Solarize_Light2')
#Solarize_Light2
#fivethirtyeight
#seaborn-pastel
#seaborn-whitegrid


#print(plt.style.available)
styles_in_plt = ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 
	'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
	'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 
	'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 
	'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 
	'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 
	'seaborn-whitegrid', 'tableau-colorblind10']



'''
fig = plt.figure()
ax = plt.axes()
#plt.savefig('fig.png')
plt.show()




x = np.linspace(0, 10, 1000)
y = np.sin(x)
fig = plt.figure()
ax = plt.axes()
ax.plot(x, y)
#plt.savefig('plot.png')
plt.show()




x = np.linspace(0, 10, 1000)
y = np.sin(x)
fig = plt.figure()
plt.plot(x, y)
plt.show()




x = np.linspace(0, 10, 1000)
#y = np.sin(x)
#fig = plt.figure()
plt.plot(x, np.cos(x))
#plt.plot(x, np.sin(x))
#plt.plot(x, np.tan(x))
plt.show()




x = np.linspace(0, 10, 1000)

#The figure size
plt.figure(figsize=(9, 6))
#function1
plt.plot(x, np.cos(x), label='cosinus')
#function2
plt.plot(x, np.sin(x), label='sinus')
#The title of graphic
plt.title('figure1')
#axe x in the graphic
plt.xlabel('axe x')
#axe y in the graphic
plt.ylabel('axe y')
plt.legend() #add the name of legend in function1 and function2
plt.show()



i = 20
x = np.linspace(0, 10, i)
y = np.random.randint(1, 50, i)

#plt.plot(y)
plt.plot(x, y, color='c', marker=True) #line plot with marker in cyan line

plt.show()



x = np.linspace(0, 10, 1000)
y = np.sin(x)

fig = plt.figure()
plt.plot(x, y)
plt.show()




x = np.linspace(0, 10, 1000)

plt.plot(x, np.sin(x))
plt.show()




x = np.linspace(0, 10, 1000)  #1darray of length 1000
y = np.sin(x)

#plt.plot(x, y)

fig, ax = plt.subplots()

ax.set_xlabel('axis \'x\'')
ax.set_ylabel('axis \'y\'')
ax.set_title('function sin(x)')

plt.show()




x = np.linspace(0, 10, 1000)  #1darray of length 1000
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('axis \'x\'')
plt.ylabel('axis \'y\'')

plt.title('function sin(x)')
#plt.legend(loc=5)

plt.show()




x = np.linspace(0, 10, 1000)
y = np.sin(x)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.xlabel('axis \'x\'')
plt.ylabel('axis \'y\'')
plt.title('function sin(x)')

plt.show()




x = np.linspace(0, 10, 1000)
y = np.sin(x)

plt.plot(x, np.sin(x), color='r')
plt.plot(x, np.cos(x), color='c', linestyle='-.')

plt.xlabel('axis \'x\'')
plt.ylabel('axis \'y\'')
plt.title('function sin(x)')

plt.show()


#b: blue
#g: green
#r: red
#c: cyan
#m: magenta
#y: yellow
#k: black
#w: white

#opacity: 'alpha=0.8'

#lines: '-'  '--'  '-.'  ':' '*'




x = np.linspace(0, 10, 1000)
y = np.sin(x)

plt.plot(x, np.sin(x), 'y-', label='sin(x)')
plt.plot(x, np.cos(x), 'r-.', label='cos(x)')
plt.legend(loc=4)

plt.xlabel('axis \'x\'')
plt.ylabel('axis \'y\'')
plt.title('function sin(x)')

plt.show()




presidents_df = pd.read_csv('president_heights_party.csv')

#plt.plot(presidents_df['height'], presidents_df['age'], 'o')
plt.scatter(presidents_df['height'], presidents_df['age'])
#plt.savefig('plot2.png')
plt.show()




presidents_df = pd.read_csv('president_heights_party.csv')

plt.scatter(presidents_df['height'], presidents_df['age'],
	marker='3',
	color='g')

plt.xlabel('height')
plt.ylabel('age')
plt.title('U.S. presidents')

#plt.savefig('plot3.png')
plt.show()


#markers: '<' '>' 's' 'x' '*' '.'




presidents_df = pd.read_csv('president_heights_party.csv')

presidents_df.plot(kind='scatter',
	x='height',
	y='age',
	title='U.S. presidents')

plt.show()




presidents_df = pd.read_csv('president_heights_party.csv')

presidents_df['height'].plot(kind='hist',
	title='height',
	bins=5,
	color='r')

plt.show()




presidents_df = pd.read_csv('president_heights_party.csv')

plt.hist(presidents_df['height'], bins=5)

plt.show()




presidents_df = pd.read_csv('president_heights_party.csv')

#print(presidents_df['height'].describe())

#plt.style.use('classic')
presidents_df.boxplot(column='height')

plt.show()




presidents_df = pd.read_csv('president_heights_party.csv')

party_cnt = presidents_df['party'].value_counts()

party_cnt.plot(kind='bar')
plt.show()




lst = [float(x) if x != 'nan' else np.NaN for x in input().split()]

lst_df = pd.Series(lst)
final = lst_df.fillna(lst_df.mean()).round(1)
print(final)
'''


'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  L I N E A R   R E G R E S S I O N   *  *  *  *  *  *
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''


#>>   y = b + m * x



'''
boston_dataset = load_boston() #build a DataFrame

boston = pd.DataFrame(boston_dataset.data,
	columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

#pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 1000)

#print(boston)
#print(boston.shape)
#print(boston.shape[0])
#print(boston.describe)
#print(boston.head(10))
#print(boston.columns)
#print(boston_dataset['target'])


#print(boston[['CHAS', 'RM', 'AGE', 'RAD', 'MEDV']].head())

#print(boston.describe().round(2))


#boston.hist(column='CHAS')
boston.hist(column='RM', bins=20)
plt.show()




boston.plot(kind='scatter',
	x='RM',
	y='MEDV',
	figsize=(8,6))

plt.show()


boston.plot(kind='scatter',
	x='LSTAT',
	y='MEDV',
	figsize=(8,6))

plt.show()


X = boston[['RM']]
print(X.shape)
print(X)

Y = boston['MEDV']
print(Y.shape)




corr_matrix = boston.corr().round(2)
print(corr_matrix)




model = LinearRegression()
print(model)


# To view all the available models we can use:
from sklearn import linear_model
lim = linear_model
print(dir(lim))




X = boston[['RM']]
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
	test_size=3,
	random_state=1)
'''
#print(X_train.shape)
#print(Y_train.shape)
#print(X_test.shape)
#print(Y_test.shape)




#from sklearn.model_selection import train_test_split

#model = LinearRegression()
#model.fit(X_train, Y_train)

#print(model.intercept_.round(2))
#print(model.coef_.round(2))




#new_RM = np.array([6.5]).reshape(-1, 1)

#print(model.predict(new_RM))
#print(model.intercept_ + model.coef_ * 6.5)


#y_test_predicted = model.predict(X_test)

#print(y_test_predicted.shape)
#print(type(y_test_predicted))
#print(y_test_predicted)
#print(Y_test.shape)


#from sklearn.metrics import r2_score
#print(r2_score(Y_test, y_test_predicted))




something_to_signpost = ['something'
						'something'
						'somethin'
						'somethi'
						'someth'
						'somet'
						'some'
						'som'
						'so'
						's'
						''
						]
'''
from sklearn.model_selection import train_test_split

model = LinearRegression()
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

X = boston[['RM']]
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
	test_size=0.3,
	random_state=1)

model.fit(X_train, Y_train)

y_test_predicted = model.predict(X_test)


plt.scatter(X_test, Y_test, label='testing data')
plt.plot(X_test, y_test_predicted, label='prediction', linewidth=3, color='g')

plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend(loc='upper left')

plt.show()
'''


'''
residuals = Y_test - y_test_predicted

 
plt.scatter(X_test, residuals)  # plot residuals
plt.hlines(y=0, xmin=X_test.min(), xmax=X_test.max(), linestyle='--')
	# plot a horizontal line at y = 0
plt.xlim((4, 9))  # set xlim
plt.xlabel('RM')
plt.ylabel('residuals')
plt.show()




#print(residuals[:5])

#print(residuals.mean())
#print(type(residuals.mean()))

#print((residuals**2).mean())  #mean squared error (MSE)

from sklearn.metrics import mean_squared_error
#print(mean_squared_error(Y_test, y_test_predicted))




#print(model.score(X_test, Y_test))

print(((Y_test - Y_test.mean())**2).sum())

print((residuals**2).sum())

print(1 - 5550.616639087471 / 13931.482039473683)




#>>   MEDV = b0 + b1 * RM + b2 * LSTAT

# data preparation
X2 = boston[['RM', 'LSTAT']]
Y = boston['MEDV']

# train test split
# same random_state to ensure the same splits
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y,
	test_size=0.3,
	random_state=1)

model2 = LinearRegression()

#print(model2.fit(X2_train, Y_train))

model2.fit(X2_train, Y_train)
#print(model2.intercept_)
#print(model2.coef_)


y_test_predicted2 = model2.predict(X2_test)
#print(y_test_predicted2)




from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, y_test_predicted).round(2))
print(mean_squared_error(Y_test, y_test_predicted2).round(2))




#import numpy as np

n, p = [int(x) for x in input().split()]
X = []

for i in range(n):
    X.append([float(x) for x in input().split()])

y = [float(x) for x in input().split()]


X = np.array(X).reshape(n,p)
y = np.array(y)
b = np.linalg.pinv(X) @ y.transpose()

print(np.around(b,decimals=2))
'''


'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  *  C L A S S I F I C A T I O N   *  *  *  *  *  *  *
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  




x1 = np.array([1, -1])
x2 = np.array([4, 3])
print(np.sqrt(((x1-x2)**2).sum()))




iris = pd.read_csv('iris.csv')
#print(iris.shape)
#print(iris.head())
#print(iris.info())
#print(iris.describe())


iris.drop('id', axis=1, inplace=True)
'''

#iris.drop('id', axis=1, inplace=True)
#print(iris.head())

#print(iris[['petal_len', 'petal_wd']].describe())

#print(iris.size())
#print(iris.groupby('species').size())

#print(iris['species'].value_counts())



'''
iris['sepal_wd'].hist()
plt.show()
'''


'''
# build a dict mapping species to an integer code
inv_name_dict = {
	'iris-setosa': 0,
	'iris-versicolor': 1,
	'iris-virginica': 2
}

# build integer color code 0/1/2
colors = [inv_name_dict[item] for item in iris['species']]

#scatter plot
scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c=colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

# add legend
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())

plt.show()
'''


'''
# build a dict mapping species to an integer code
inv_name_dict = {
	'iris-setosa': 0,
	'iris-versicolor': 1,
	'iris-virginica': 2
}

# build integer color code 0/1/2
colors = [inv_name_dict[item] for item in iris['species']]

#scatter plot
scatter = plt.scatter(iris['petal_len'], iris['petal_wd'], c=colors)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

# add legend
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())

plt.show()
'''


'''
#pd.plotting.scatter_matrix(iris)

x = np.linspace(0, 8, 1000)
y1 = [4, 7, 9]
y2 = np.sin(x)
y3 = np.arcsin(x)

line, = plt.plot(y1)
sine, = plt.plot(x, y2)
arcsine, =plt.plot(x, y3)

plt.legend(handles=[line, sine, arcsine],
	labels=['Line', 'Sine', 'Arcsine'])

plt.show()
'''


'''
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], c=iris.target, marker='.')
plt.show()
'''


'''
from sklearn import datasets

iris_dataset = datasets.load_iris()
for i in iris_dataset:
	print(i)

iris = pd.DataFrame(data=iris_dataset['data'],
	columns=iris_dataset['feature_names'])
t = list(iris_dataset.target_names)
print(type(t))

target = []
for i in iris_dataset.target:
	target.append(t[i])

iris['target'] = target
print(iris.columns)

scatter = plt.scatter(iris['sepal length (cm)'],
	iris['sepal width (cm)'], c=iris_dataset.target)

print(scatter.legend_elements())

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend(handles=scatter.legend_elements()[0], labels=t)
plt.show()
'''


'''
X = iris[['petal_len', 'petal_wd']]
y = iris['species']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
	random_state=1, stratify=y)


#print(y_train.value_counts())
#print(y_test.value_counts())


from sklearn.neighbors import KNeighborsClassifier
'''

'''
## instantiate
knn = KNeighborsClassifier(n_neighbors=5)


## fit
#print(knn.fit(X_train, y_train))
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
#print(pred[:5])


y_pred_prob = knn.predict_proba(X_test)

print(y_pred_prob[10:12])

print(pred[10:12])


print((pred==y_test.values).sum())
print(y_test.size)
print()
print((pred==y_test.values).sum() / y_test.size)
print()
print(knn.score(X_test, y_test))
'''


'''
from sklearn.metrics import confusion_matrix

#print(confusion_matrix(y_test, pred))


from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)

plt.show()
'''


'''
from sklearn.model_selection import cross_val_score

## create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=4)

## train model with 5-fold cv
cv_scores = cross_val_score(knn_cv, X, y, cv=10)

## print each cv score (accuracy)
print(cv_scores.round(4))

## then average them
print(cv_scores.mean())
'''


'''
from sklearn.model_selection import GridSearchCV

## create a new knn model
knn2 = KNeighborsClassifier()

## create a dict of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}

## use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

## fit model to data
knn_gscv.fit(X, y)

#print(knn_gscv.best_params_)
#print(knn_gscv.best_score_)
'''


'''
knn_final = KNeighborsClassifier(n_neighbors=
	knn_gscv.best_params_['n_neighbors'])

knn_final.fit(X, y)

#y_pred = knn_final.predict(X)
#print(knn_final.score(X, y))




#new_data = np.array([3.76, 1.20])
#new_data = new_data.reshape(1, -1)
#print(knn_final.predict(new_data))

## same result
print(knn_final.predict([[3.76, 1.20]]))
'''


'''
new_data = np.array([[3.76, 1.2],
					 [5.25, 1.2],
					 [1.58, 1.2]])
#print(knn_final.predict(new_data))

print(knn_final.predict_proba(new_data))
'''


'''
from sklearn.metrics import confusion_matrix

y_true = np.array(['cat', 'dog', 'dog', 'cat', 'fish', 'dog', 'fish'])
y_pred = np.array(['cat', 'cat', 'cat', 'cat', 'fish', 'dog', 'fish'])

confusion_matrix(y_true, y_pred, labels=['cat', 'dog', 'fish'])

print(confusion_matrix(y_true, y_pred))
'''


'''
from sklearn.metrics import confusion_matrix

y_true = [int(x) for x in input().split()]
y_pred = [int(x) for x in input().split()]


y_true = np.array(y_true)
y_pred = np.array(y_pred)
print(confusion_matrix(y_pred, y_true, labels=[True, False]).astype(float))




s = confusion_matrix(y_true, y_pred)
a = np.flip(s)
s[0][0], s[-1][-1] = s[-1][-1], s[0][0]
s = s.astype(float)
print(s)
'''


'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  *  *   C L U S T E R I N G    *  *  *  *  *  *  *  *
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''


'''
x1 = np.array([0, 1])
x2 = np.array([2, 0])

x1 = np.array([1, -1])		# (p1, p2)
x2 = np.array([4, 3])		# (q1, q2)
							# sqrt( (p1 - q1)**2 + (p2 - q2)**2 )

print(np.sqrt(((x1 - x2)**2).sum()))
'''



from sklearn.datasets import load_wine

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)

#print(wine.shape)
#print(wine.columns)
#print(wine.info)

#print(wine.iloc[:,:3].describe())



'''
from pandas.plotting import scatter_matrix

scatter_matrix(wine.iloc[:, [0, 5]])
#scatter_matrix(wine.iloc[:, [0, 1, 2, 3, 4]], figsize=(11, 11))
#scatter_matrix(wine.iloc[:, [5, 6]])
#scatter_matrix(wine.iloc[:, [0, 12]])
plt.show()
'''



#X = wine[['alcohol', 'total_phenols']]
X = wine

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scale.fit(X)

#print(scale.mean_)
#print(scale.scale_)


X_scaled = scale.transform(X)
#print(X_scaled.mean(axis=0))
#print(X_scaled.std(axis=0))

'''
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], marker='.', label='scaled data')
plt.scatter(X.iloc[:, [0]], X.iloc[:, [1]], marker='*', label='original data')
plt.legend()
plt.show()
'''



from sklearn.cluster import KMeans
'''
## instantiate the model
kmeans = KMeans(n_clusters=3)

## fit the model
kmeans.fit(X_scaled)

## make predictions
y_pred = kmeans.predict(X_scaled)
#kmeans.predict(X_scaled)
#print(y_pred)


kmeans.predict(X_scaled)
print(kmeans.cluster_centers_)
'''


'''
## plot the scaled data
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred)

## identify the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
	marker='*',
	s=250,
	c=[0, 1, 2],
	edgecolor='k')

plt.xlabel('alcohol')
plt.ylabel('total phenols')
plt.title('k means (k=3)')
plt.show()
''' 


'''
X_new = np.array([[13, 2.5]])
X_new_scaled = scale.transform(X_new)

print(kmeans.predict(X_new_scaled))
'''


'''
#print(kmeans.inertia_)

## calculate distortion for a range of number of cluster
inertia = []
for i in np.arange(1, 11):
	km = KMeans(n_clusters=i)
	km.fit(X_scaled)
	inertia.append(km.inertia_)

## plot
plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
'''


'''
k_opt = 3
kmeans = KMeans(k_opt)
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)
print(y_pred)
'''




'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  L A S T   C O D E   P R O J E C T   T E S T S   *  *  *  *
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''




'''
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


scale = StandardScaler()
X = np.array([[0, 0], [2, 2]], dtype='float64')
scale.fit(X)
X_scaled = scale.transform(X)
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X_scaled)


first = np.array([[0, 0]])
second = np.array([[2, 2]])


for i in range(int(input())):
	lst = [input().split()]
	X_new = np.array(lst, dtype='float64')
	X_new_scaled = scale.transform(X_new)
	p = kmeans.predict(X_new_scaled)
	if 0 in p:
		first = np.concatenate((first, X_new), axis=0)
	elif 1 in p:
		second = np.concatenate((second, X_new), axis=0)


if len(first) == 1:
	print('None')
else:
	F = first[1:,].mean(axis=0)
	print(np.around(F, 2))


if len(second) == 1:
	print('None')
else:
	S = second[1:,].mean(axis=0)
	print(np.around(S, 2))
'''



'''
import numpy as np
from sklearn.cluster import KMeans


n = int(input())
list1 = []
list2 = []
matrix = []

for i in range(n):
	matrix.append(input().split())


X = np.array([[0, 0], [2, 2]], dtype='float64')
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
j = 0


for i in kmeans.predict(matrix):
	if i == 0:
		list1.append(matrix[j])
	if i == 1:
		list2.append(matrix[j])
	j+=1


res1 = np.round(np.array(list1).astype(float), 2)
res2 = np.round(np.array(list2).astype(float), 2)  


if len(list1) == 0:
	print('None')
else: 
	kmeans1 = KMeans(n_clusters=1, random_state=0).fit(res1)
	print(np.round(np.array(kmeans1.cluster_centers_[0]).astype(float), 2))

if len(list2) == 0:
	print('None')
else:
	kmeans2 = KMeans(n_clusters=1, random_state=0).fit(res2)
	print(np.round(np.array(kmeans2.cluster_centers_[0]).astype(float), 2))
'''



'''
import numpy as np


first = np.array([[0., 0.]])
second = np.array([[2., 2.]])
n = int(input())

data = []

for i in range(n):
	data.append([float(i) for i in input().split()])


data = np.array(data).reshape((-1,2))


for i in range(n):
	dist1 = np.sqrt(((data[i]-first[0])**2).sum())
	dist2 = np.sqrt(((data[i]-second[0])**2).sum())

	if (dist1) <= (dist2):
		first = np.vstack((first,data[i]))
	else:
		second = np.vstack((second,data[i]))


if first.size > 2:
	mean1 = np.mean(first[1:], axis=0)
	print(np.around(mean1, decimals=2))
else:
	print(None)


if second.size > 2:
	mean2 = np.mean(second[1:], axis=0)
	print(np.around(mean2, decimals=2))
else:
	print(None)
'''