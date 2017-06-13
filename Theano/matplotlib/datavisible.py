import matplotlib.pyplot as plt
import numpy as np
list = [1,2,4,5,6,7,12,25,111]
year = np.array(list)
print year
population = year**2 + 1
print population
plt.plot(year, population)
plt.xlabel('year')
plt.ylabel('population')
plt.title('world population Projections')
plt.yticks([0,2,4,6,8,10],['0','2B','4B','6B','8B','10B'])
plt.show()