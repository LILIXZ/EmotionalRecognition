from builtins import range
import numpy as np
from numpy import asarray
import os
from imageio import imread
import platform
from PIL import Image
from matplotlib.image import imread
import random

def load_FER2013_batch(dir_name, filename):
    """ load single batch of FER-2013 """
    cate_dict = {'angry': 0,'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    img = Image.open(filename).convert('RGB')
    X = np.array(img)
    # X = X.reshape((-1, 48, 3, 48)) 
    X = X.reshape((-1, 48, 48, 3)) #[48, 48, 3]
    X = X.astype("float")
        
    Y = cate_dict[dir_name]
    Y = np.array([Y]).astype("float")
    return X, Y
      
def load_FER2013(ROOT):
    """ load all of FER-2013 """
    train_path = os.path.join(ROOT, 'train')
    test_path = os.path.join(ROOT, 'test')
    
    i_list = []
    x_train_dict = {}
    y_train_dict = {}
    x_train_list = []
    y_train_list = []
    
    j_list = []
    x_val_dict = {}
    y_val_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []
    
    i = 0
    for dir in os.scandir(train_path):
      if dir.is_dir():
        for root, dirs, files in os.walk(dir):
          for file in files:
            f = os.path.join(root, file)
            X, Y = load_FER2013_batch(dir.name, f)
            x_train_dict["data_"+str(i)] = X
            y_train_dict["label_"+str(i)] = Y
            i_list.append(i)
            i = i+1
        
    random.shuffle(i_list)
    
    for k in i_list:
      x_train_list.append(x_train_dict["data_"+str(k)])
      y_train_list.append(y_train_dict["label_"+str(k)])
    
    Xtr = np.concatenate(x_train_list)
    Ytr = np.concatenate(y_train_list)
      
    del X, Y
    
    j = 0
    for dir in os.scandir(test_path):
      if dir.is_dir():
        for root, dirs, files in os.walk(dir):
          for file in files:
            f = os.path.join(root, file)
            X, Y = load_FER2013_batch(dir.name, f)
            x_test_dict["data_"+str(j)] = X
            y_test_dict["label_"+str(j)] = Y
            j_list.append(j)
            j=j+1
    
    random.shuffle(j_list)
    
    ki = 0
    for k_2 in j_list:
      if ki < 100:
        x_val_list.append(x_test_dict["data_"+str(k_2)])
        y_val_list.append(y_test_dict["label_"+str(k_2)])
        
      else:
        x_test_list.append(x_test_dict["data_"+str(k_2)])
        y_test_list.append(y_test_dict["label_"+str(k_2)])
      ki = ki + 1
      
    Xval = np.concatenate(x_val_list)
    Yval = np.concatenate(y_val_list)
    Xte = np.concatenate(x_test_list)
    Yte = np.concatenate(y_test_list)
    del X, Y
    return Xtr, Ytr, Xval, Yval, Xte, Yte
  
def get_FER2013_data(
    num_training=27000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    Load the FER-2013 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw FER-2013 data
    cifar10_dir = os.path.join(
        os.path.dirname(__file__), "archive"
    )
    X_train, y_train, X_val, y_val, X_test, y_test = load_FER2013(cifar10_dir)
    
    # Subsample the data
    # mask = list(range(num_training, num_training + num_validation))
    # X_val = X_train[mask]
    # y_val = y_train[mask]
    # mask = list(range(num_training))
    # X_train = X_train[mask]
    # y_train = y_train[mask]
    # mask = list(range(num_test))
    # X_test = X_test[mask]
    # y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
