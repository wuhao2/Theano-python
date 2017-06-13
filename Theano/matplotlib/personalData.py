import matplotlib.pyplot as plt
import numpy as np
list = [1,2,4]
year = np.array(list)
population = year**2
# add data
year = [1800,1850,1900] + year
population = [1.0, 1.262, 1.650] + population

plt.fill_between(year,population,0,color='green')

plt.plot(year, population)
plt.xlabel('year')
plt.ylabel('population')
plt.title('world population Projections')
plt.yticks([0,2,4,6,8,10],['0','2B','4B','6B','8B','10B'])
plt.show()