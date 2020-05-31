from keras.layers import MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, SeparableConv2D, BatchNormalization
from keras.models import Sequential
from keras.layers.core import Activation
from numpy import load
from keras import backend as K
from livelossplot.keras import PlotLossesCallback
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

num_classes = 2
img_width, img_height, channels = 48, 48, 3
batch_size = 32
epochs = 25

def buildModel(width, height, depth, classes):
    model = Sequential()
    shape = (height,width,depth)
    channelDim = -1
    if K.image_data_format() == "channels_first":
        shape = (depth,height,width)
        channelDim = 1
    model.add(SeparableConv2D(32, (3, 3), padding = 'same', input_shape = shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(SeparableConv2D(64, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = channelDim))
    model.add(SeparableConv2D(64, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = channelDim))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(SeparableConv2D(128, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = channelDim))
    model.add(SeparableConv2D(128, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = channelDim))
    model.add(SeparableConv2D(128, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = channelDim))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model

model = buildModel(width=48, height=48, depth=3, classes=2)

model.summary()

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'sgd', 
              metrics = ['accuracy'])

train_data = load('train_data.npy')
train_targets = load('train_targets.npy')
test_data = load('test_data.npy')
test_targets = load('test_targets.npy')
valid_data = load('valid_data.npy')
valid_targets = load('valid_targets.npy')

model.fit(train_data, train_targets,
          batch_size = batch_size,
          epochs = epochs,
          callbacks = [PlotLossesCallback()],
          verbose = 1,
          validation_data = (valid_data, valid_targets))

"""score = model.evaluate(test_data, test_targets, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])"""

y_prediction = model.predict(test_data)

accuracy = accuracy_score(y_true = test_targets, y_pred = y_prediction)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\n")
print(classification_report(test_targets, y_prediction))
