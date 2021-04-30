#!/usr/bin/env python
# coding: utf-8

# In[1]:


#How to create one dimensional NumPy array?


# In[2]:


import numpy as np # import numpy package
one_d_array = np.array([1,2,3,4]) # create 1D array


# In[3]:


print(one_d_array) # printing 1d array


# In[5]:


#How to create two dimensional NumPy array?


# In[4]:


import numpy as np # impoer numpy package
two_d_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # create 1D array
print(two_d_array) #printing 2D array


# In[7]:


#How to check the type of ndarray?


# In[8]:


type(one_d_array) # give the type of data present in one_d_array variable


# In[9]:


#How to check dimension of NumPy ndarray?
#The ndim attribute help to find the dimension of any NumPy array.


# In[10]:


one_d_array.ndim # find the dimension of one_d_array


# In[11]:


#How to check the size of the NumPy array?
#The size attribute help to know, how many items present in a ndarray.


# In[12]:


one_d_array.size


# In[13]:


#How to check the shape of ndarray?
#The shape attribute help to know the shape of NumPy ndarray. It gives output in the form of a tuple data type. Tuple represents the number of rows and columns. Ex: (rows, columns)


# In[14]:


two_d_array.shape


# In[15]:


#How to the data type of NumPy ndarray?
#The dtype attribute help to know the data type of ndarray.


# In[17]:


one_d_array.dtype


# In[18]:


#Create metrics using python NumPy functions 
#Ones metrics use NumPy ones() function.
#Syntax: np.ones(shape, dtype=None, order=‘C’)


# In[19]:


np.ones((3,3), dtype = int)


# In[20]:


np.empty((2,4))


# In[21]:


#Create NumPy 1D array using arange() function


# In[22]:


arr = np.arange(1,13)
print(arr)


# In[23]:


#Create NumPy 1D array using linspace() function
#Return evenly spaced numbers over a specified interval.
#Syntax: np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0,)


# In[24]:


np.linspace(1,5,4)


# In[25]:


#Convert 1D array to multidimensional array using reshape() function
#Syntax: np.reshape(a, newshape, order=‘C’)


# In[26]:


arr_reshape = np.reshape(arr, (3,4))
print(arr_reshape)


# In[27]:


#Convert multidimensional array in one dimensional
#To convert multidimensional array into 1D use ravel() or flatten() function.
#Syntax: np.ravel(array_name, order=‘C’)  or  array_name.ravel(order=‘C’)
# array_name.flatten(order=‘C’)


# In[29]:


arr_reshape.flatten()


# In[32]:


arr


# In[35]:


x=arr
y=arr_transpose = np.transpose(arr)
y


# In[36]:


x= np.array([[1, 2, 3], [4, 5, 6]])
y=arr_transpose = np.transpose(arr)
y


# In[37]:


arr1 = np.arange(1,10).reshape(3,3)
arr2 = np.arange(1,10).reshape(3,3)
 


# In[38]:


print(arr1)
print(arr2)


# In[39]:


#Addition of Two Numpy Array


# In[40]:


arr1 + arr2


# In[41]:


np.add(arr1, arr2)


# In[42]:


#Subtraction of Two NumPy Array


# In[43]:


arr1 - arr2


# In[44]:


np.subtract(arr1, arr2)


# In[45]:


arr1 / arr2


# In[46]:


np.divide(arr1, arr2)


# In[47]:


#Multiplication of Two NumPy Array


# In[48]:


arr1 * arr2


# In[49]:


#Using np.multiply() function


# In[50]:


np.multiply(arr1, arr2)


# In[51]:


#Matrix Product of Two NumPy Array (matrix)
#using @ Operator


# In[52]:


arr1 @ arr2


# In[53]:


arr1.dot(arr2)


# In[54]:


#NumPy Mathematical Built-in functions


# In[55]:


arr1.max()


# In[56]:


arr1.max(axis = 0) # return max value from each column


# In[57]:


arr1.max(axis = 1) # return max value from each row


# In[58]:


arr1.argmax() # return index of max value of an array


# In[59]:


arr1.min()


# In[60]:


arr1.min(axis = 0) # return min value from each column


# In[62]:


arr1.argmin()


# In[63]:


np.sum(arr1)


# In[64]:


np.sum(arr1, axis = 0)


# In[65]:


np.sum(arr1, axis = 1)


# In[66]:


np.mean(arr1)


# In[67]:


np.sqrt(arr1)


# In[68]:


np.std(arr1)


# In[69]:


np.exp(arr1)


# In[70]:


np.log(arr1)


# In[71]:


np.log10(arr1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




