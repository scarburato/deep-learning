import tensorflow as tf
from common import *

model = tf.keras.models.load_model(sys.argv[3])

plot(model, None, X_test, Y_test, val_dataset, "USCITA_FINALE")