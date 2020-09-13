import matplotlib.pyplot as plt

colors = (0,0,0)

x = [2.99,2.99,3.5,3.7,3.9,3.9,3.5,3.5,2.5,3,3.1]                                               
y = [17,18,23,26,26,26,18,19,8,21,21]  

plt.scatter(x, y, s=10, c=colors, alpha=0.5) 
plt.title('Scatter plot Subway Sandwitches') 
plt.xlabel('Cost') 
plt.ylabel('Protein') 
plt.show()     
