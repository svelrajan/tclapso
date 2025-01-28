import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import sys

from numpy import genfromtxt
data_file = sys.argv[1]

my_data = genfromtxt(data_file, delimiter=',')
my_data = my_data.transpose()
my_data = np.nan_to_num(my_data)

p_data = pd.DataFrame(my_data)
tp_data = p_data.transpose()
print(tp_data)
tp_data = tp_data.sort_values(4, ascending=True)
print(tp_data[7])

print("*** Sessions ****")
print(tp_data[7])
print("*** Migration Time ****")
print(tp_data[12])

x = tp_data[7].to_numpy()
y = tp_data[12].to_numpy()

n = np.size(x)

x = x.reshape(-1,1)
regression_model = LinearRegression()
  
# Fit the data(train the model)
regression_model.fit(x, y)
  
# Predict
y_predicted = regression_model.predict(x)
  
# model evaluation
mse=mean_squared_error(y,y_predicted)
  
rmse = np.sqrt(mean_squared_error(y, y_predicted))
r2 = r2_score(y, y_predicted)
  
# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('MSE:',mse)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

plt.xlim(0, max(x))
plt.ylim(0, max(y) + 500)

print("Max values")
print(max(x))
print(max(y))
plt.scatter(x, y)
plt.plot(x, y_predicted, '-r', label = "Migration Time")
plt.title('Sessions vs. Migration Time')
plt.xlabel('Sessions', color='#1C2833')
plt.ylabel('Migration Time (in msecs)', color='#1C2833')
plt.legend(loc='upper left')

new_file = data_file + "-sess-vs-mt" + ".png"
plt.savefig(new_file)
plt.grid()
#plt.show()
