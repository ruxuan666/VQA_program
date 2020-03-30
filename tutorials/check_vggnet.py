#查看并修改VGGnet
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models
import torch.nn as nn


# In[2]:

model = models.vgg16(pretrained=True)
print(model)


# In[3]:


model.classifier = nn.Sequential(
    *list(model.classifier.children())[:-1])    # remove last fc layer
print(model)


# In[ ]:




