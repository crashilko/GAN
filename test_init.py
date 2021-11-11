from gan_model import *
import matplotlib.pyplot as plt

model = GANModel('cross_entropy', 'Adam', 'cross_entropy', 'Adam')
image = model.generate_image()
plt.imshow(image[0,:,:,0])
plt.show()