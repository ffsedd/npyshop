from npimage import npImage
import nputils
from matplotlib import pyplot as plt

img = npImage("sample2.jpg")

print(img)
#print(img.stats)
plt.imshow(img.arr, cmap='jet',interpolation=None)
plt.show()


#img.color_model = 'hsv'
img.arr = nputils.gamma(arr=img.arr, g=.5)
print(img)
#print(img.stats)
plt.imshow(img.arr, cmap='jet',interpolation=None)


plt.show()

img.color_model = 'rgb'
#img.arr = nputils.gamma(arr=img.arr, g=.5)
print(img)
#print(img.stats)
plt.imshow(img.arr, cmap='jet',interpolation=None)

plt.show()