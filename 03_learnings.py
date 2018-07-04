import numpy as np
# lets learn how split works
array1=[1,2,3,4,5,6,7,8,9]
array5=np.array([[1,2,3],[4,5,6],[6,7,8],[8,9,0],[4,5,8]])
array2=array1[:]
array3=array1[2:5]
array4=array5[:,2]
print(array1)
print(array2)
print(array3)
print(array4)

# A numpy array is totally different when comapred to normal array
# arr_numpy =[[1 2 3] [3 4 5]]
# arr_nrml =[[1,2,3][3,4,5]]
# while using above segregation things work out only when it is numpy array
