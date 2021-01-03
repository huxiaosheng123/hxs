import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data2_loss =np.loadtxt("F:/log.txt")
data1_loss = np.loadtxt("F:/log11.txt")

#x = data1_loss[:,0]
x = np.arange(30)
print(x)
y = data1_loss[:]
print(y)
#x1 = data2_loss[:,0]
x1 = np.arange(30)
y1 = data2_loss[:]
print(x1)
print(y1)
fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`

# plt.gcf().set_facecolor(np.ones(4)* 240 / 255)   # 生成画布的大小
plt.grid()  # 生成网格
#y = np.linspace(-3,3)  #设置横轴的取值点
pl.plot(x,y,'r-',label=u'nuetral_prior')
# ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
p2 = pl.plot(x1, y1,'b-', label = u'no_prior')
pl.legend()
#显示图例
#p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend()
pl.xlabel(u'epoch')
pl.ylabel(u'loss')
plt.title('Compare loss for different models in training')
plt.show()