#做结果图
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import csv
import matplotlib.pyplot as plt


# In[ ]:


num_epochs = 18


# In[ ]:


fig = plt.figure(figsize=(10,5))
#分别做训练集与验证集的线
for phase in ['train', 'valid']:
    
    epoch = []
    loss = []
    acc = []
    
    for i in range(num_epochs):
        
        with open('./logs/{}-log-epoch-{:02d}.txt'.format(phase, i+1), 'r') as f:
            df = csv.reader(f, delimiter='\t') #只有一行
            data = list(df)#转换为列表，[[每行元素]]

        epoch.append(float(data[0][0]))
        loss.append(float(data[0][1]))
        acc.append(float(data[0][3]))

    plt.subplot(1, 2, 1)# 1*2的画布
    if phase == 'train':
        plt.plot(epoch, loss, label = phase, color = 'red', linewidth = 5.0)
    else:
        plt.plot(epoch, loss, label = phase, color = 'blue', linewidth = 5.0)
            
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
        
    plt.subplot(1, 2, 2)
    plt.tight_layout()

    if phase == 'train':
        plt.plot(epoch, acc, label = phase, color = 'red', linewidth = 5.0)
    else:
        plt.plot(epoch, acc, label = phase, color = 'blue', linewidth = 5.0)
    
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    #线标签设置
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 20})

plt.savefig('./png/train1.png', dpi = fig.dpi)

