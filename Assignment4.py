# importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
%matplotlib inline
data = pd.read_csv('mnist.csv')
data.head()
a = data.iloc[3,1:].values
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)
<matplotlib. image.AxesImage at 0x7fdd8a3cla90>
df_x = data.iloc[:, 1:]
df_y = data.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)
x [train.head( )
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
pred
array([0, 4, 0, ... , 0, 3, 1])
s = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count+1
count
len(pred)  
