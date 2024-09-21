import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   sklearn.linear_model import LinearRegression
from   sklearn.model_selection import train_test_split

#Generate the data
data={
    'squarefootage':[1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'bedrooms':[3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    'bathrooms':[2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'prices':[300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}
df= pd.DataFrame(data)
X=df[['squarefootage','bedrooms','bathrooms']]
y=df['prices']

#Split the data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#Initialize and train the linear regression model
model=LinearRegression()
model.fit(X_train,y_train)

#Make predictions
y_pred=model.predict(X_test)

#visualize prediction vs actual prices
plt.scatter(y_test,y_pred)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("Actual vs Predicted Price")
plt.show()