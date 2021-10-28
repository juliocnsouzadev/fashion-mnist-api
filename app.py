from scipy.misc import imsave
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imsave(name="images/{}.png".format(i), arr=X_test[i])
