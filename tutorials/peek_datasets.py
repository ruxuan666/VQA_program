#查看数据
#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import io


# In[2]:
vqa = np.load('../datasets/train.npy',allow_pickle=True)


# In[3]:
print(len(vqa))


# In[4]:
idx = 0


# In[5]:
print(vqa[idx].keys())


# In[6]:
print(vqa[idx])


# In[7]:
plt.figure()
image = io.imread(vqa[idx]['image_path'])
print(image)#三元数组（长，宽，3），0-255，转换为RGB数据
print(image.shape)
plt.imshow(image)
plt.show()


# In[8]:


print(vqa[idx]['all_answers'])


# In[9]:

print(vqa[idx]['valid_answers'])


# In[10]:

print(len(vqa[idx]['valid_answers']))
