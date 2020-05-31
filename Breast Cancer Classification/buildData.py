from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile
from numpy import save

num_classes = 2
img_width, img_height, channels = 48, 48, 3

def load_dataset(path, shuffle):
    data = load_files(path,shuffle=shuffle)
    condition_files = np.array(data['filenames'])
    print(len(condition_files))
    condition_targets = np_utils.to_categorical(np.array(data['target']), num_classes)
    return condition_files, condition_targets

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# load train, test, and validation datasets
train_files, train_targets = load_dataset('training', True)
valid_files, valid_targets = load_dataset('validation', True)
test_files, test_targets = load_dataset('testing', True)

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

save('train_data.npy', train_tensors)
save('train_targets.npy', train_targets)
save('test_data.npy', test_tensors)
save('test_targets.npy', test_targets)
save('valid_data.npy', valid_tensors)
save('valid_targets.npy', valid_targets)