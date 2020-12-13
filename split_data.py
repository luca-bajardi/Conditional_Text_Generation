import numpy as np
import math
from sklearn.model_selection import train_test_split

xy_train_validate = np.load('xy_train_validate.npy')
xy_test = np.load('xy_test.npy')
l1 = list(xy_train_validate)
l2 = list(xy_test)
all_xy = l1+l2

def split_in_two(s):
    words = s.split()
    split = int(math.ceil(len(words)/2.0))
    return " ".join(words[:split])," ".join(words[split:])


all_x = []
all_y = []
for s in all_xy:
    first_half, second_half = split_in_two(s)
    all_x.append(first_half)
    all_y.append(second_half)

X = all_x
Y = all_y

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val = np.array(x_val)
y_val = np.array(y_val)

x_test = np.array(x_test)
y_test = np.array(y_test)

np.save('x_train',x_train)
np.save('y_train',y_train)

np.save('x_val',x_val)
np.save('y_val',y_val)

np.save('x_test',x_test)
np.save('y_test',y_test)
