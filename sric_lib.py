import os
import cv2
import random
import shutil
import time
import matplotlib
import glob
import operator
import psutil
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from shutil import copyfile
import pandas as pd
from tqdm import tqdm 
import PIL
from mlxtend.plotting import plot_confusion_matrix
from contextlib import redirect_stdout


physical_devices = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.list_physical_devices()
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if physical_devices != []:
    print("Using GPU")
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
else:
    print("Using CPU")
    pass


def analyze(source_address):
    source_folder_dir = os.path.join(source_address, 'training')

    source_data = sum([len(files) for r, d, files in os.walk(source_folder_dir)])
    print("Total no. of files: {}".format(source_data))


    class_no = len(os.listdir(source_folder_dir))

    classes = []
    source_folder_dir_class = []

    files = {}
    for i in range(0,class_no):
        i += 1
        classes = os.listdir(source_folder_dir)

        source_folder_dir_class.append(os.path.join(source_folder_dir, classes[i-1]))
        
        for filename in os.listdir(source_folder_dir_class[i-1]):
            file = os.path.join(source_folder_dir_class[i-1], filename)
            x = os.path.getsize(file) 
            files.update({file:x})
        
        sorted_files = sorted(files.items(), key=operator.itemgetter(1))
        img_src = sorted_files[0][0]

        image = cv2.imread(img_src)
        smallest_h = image.shape[0]
        smallest_w = image.shape[1]

    return source_data, class_no, classes, source_folder_dir_class, smallest_h, smallest_w



def mk_splitted_dir(source_address):
    directory = os.path.join(source_address, 'classify train')
    train_directory = os.path.join(directory, 'training')
    validation_directory = os.path.join(directory, 'validation')
    test_directory = os.path.join(directory, 'testing')

    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        shutil.rmtree(directory)
        os.mkdir(directory)

    os.mkdir(train_directory)
    os.mkdir(validation_directory)
    os.mkdir(test_directory)
    
    return directory, train_directory, validation_directory, test_directory



def split_data(SOURCE, TRAINING, VALIDATION, TESTING, VALIDATION_SPLIT_SIZE, TEST_SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = os.path.join(SOURCE, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring!")

    validation_length = int(len(files) * VALIDATION_SPLIT_SIZE)
    testing_length = int(len(files) * TEST_SPLIT_SIZE)
    training_length = len(files) - (validation_length + testing_length)
    
    shuffled_set = random.sample(files, len(files))
    
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[(training_length):(training_length+validation_length)]
    testing_set = shuffled_set[(training_length+validation_length):(training_length+validation_length+testing_length)]

    for filename in training_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TRAINING ,filename)
        copyfile(this_file, destination)
        
    for filename in validation_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(VALIDATION, filename)
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TESTING, filename)
        copyfile(this_file, destination)



def splitted(class_no, classes, train_directory, validation_directory, test_directory, source_folder_dir_class, val_size, test_size):
    print("\nSplitting data to training-validation-testing directory...")
    train_class = []
    validation_class = []
    test_class = []

    with tqdm(total=class_no) as pbar:
        for i in range(0,class_no):
            i += 1

            train_class.append(os.path.join(train_directory, classes[i-1]))
            validation_class.append(os.path.join(validation_directory, classes[i-1] ))
            test_class.append(os.path.join(test_directory, classes[i-1]))

            os.mkdir(train_class[i-1])
            os.mkdir(validation_class[i-1])
            os.mkdir(test_class[i-1])

            split_data(source_folder_dir_class[i-1], train_class[i-1], validation_class[i-1], test_class[i-1], val_size, test_size)

            pbar.set_description("Progress")
            pbar.update()



def get_size(directory):
    print("\n\nGetting the shape of the images...")
    height = []
    width = []

    total = sum([len(files) for r, d, files in os.walk(directory)])

    with tqdm(total=total) as pbar:
        for folder_name_1 in os.listdir(directory):
            folder_1 = os.path.join(directory, folder_name_1)
            for folder_name_2 in os.listdir(folder_1):
                folder_2 = os.path.join(folder_1, folder_name_2)
                for classes_name in os.listdir(folder_2):
                    file = os.path.join(folder_2, classes_name)
                    img = cv2.imread(file)
                    height.append(img.shape[0])
                    width.append(img.shape[1])

                    pbar.set_description("Progress")
                    pbar.update()
                    
    height = np.array(height)
    width = np.array(width)
                    
    return height, width



def resize(folder, w, h):
    print("\nPre-processing the splitted data...")
    total = sum([len(files) for r, d, files in os.walk(folder)])
    with tqdm(total=total) as pbar:
        for folder_name_1 in os.listdir(folder):
            folder_1 = os.path.join(folder, folder_name_1)
            for folder_name_2 in os.listdir(folder_1):
                folder_2 = os.path.join(folder_1, folder_name_2)
                for classes_name in os.listdir(folder_2):
                    file = os.path.join(folder_2, classes_name)
                    
                    img = cv2.imread(file)
                    res = cv2.resize(img, (w, h))
                    
                    cv2.imwrite(file, res)
                    
                    dim = [h, w, 3]
                    
                    pbar.set_description("Progress")
                    pbar.update()
    return dim


        
def monitor_dir(now, name_res_dir, source_address):
    # now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    char_subfolder_name = name_res_dir
    char_subfolder = os.path.join(source_address, char_subfolder_name)

    if not os.path.exists(char_subfolder):
        os.mkdir(char_subfolder)
        char_name = dt_string

        char = os.path.join(source_address, char_subfolder, char_name)

        if not os.path.exists(char):
            os.mkdir(char)
        else:   
            shutil.rmtree(char)
            os.mkdir(char)

    else:
        char_name = dt_string

        char = os.path.join(source_address, char_subfolder, char_name)

        if not os.path.exists(char):
            os.mkdir(char)
        else:   
            shutil.rmtree(char)
            os.mkdir(char)
            
    return char, char_subfolder, char_name



def lr_schedule(epoch, learning_rate):
    return learning_rate * (0.1 ** int(epoch / 10))



def monitor_metric(custom, early_stop, monitor, patience, char):
    if monitor == 'Validation Acccuracy':
        metric = 'val_accuracy'
        mode = 'max'
        print("\nMONITORING VALIDATION ACCURACY..........\n")

    elif monitor == 'Validation Loss':
        metric = 'val_loss'
        mode = 'min'
        print("\nMONITORING VALIDATION LOSS..........\n")

    elif monitor == 'Training Accuracy':
        metric = 'accuracy'
        mode = 'max'
        print("\nMONITORING TRAINING ACCURACY..........\n")

    elif monitor == 'Training Loss':
        metric = 'loss'
        mode = 'min'
        print("\nMONITORING TRAINING LOSS..........\n")
        
    best_model_address = os.path.join(char, 'best_model.h5')
        
    callback = [keras.callbacks.LearningRateScheduler(lr_schedule, verbose = 1),        
                keras.callbacks.ModelCheckpoint(best_model_address, monitor = metric, verbose=1, save_best_only=True, save_weights_only=False, mode = mode),
                keras.callbacks.EarlyStopping(monitor = metric, min_delta = 0.001, patience = patience, verbose=1, mode = mode, restore_best_weights = True)]

        
    if custom.upper() == 'Y':
        if early_stop.upper() == 'Y':
            callback = callback
        elif early_stop.upper() == 'N':
            callback.pop(-1)
            
    elif custom.upper() == 'N':
        metric = 'val_accuracy'
        mode = 'max'
        patience = 10

        callback = callback

    return callback, best_model_address



def vgg16(train_base_model, dim, dense, dropout, output_layer, output_activation):
    print("\nTRAINING ON VGG16 MODEL:-")

    base_model = keras.applications.vgg16.VGG16(input_shape = dim, weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(dense)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    predictions = Dense(output_layer, activation = output_activation)(x)

    model = Model(inputs = base_model.input, outputs=predictions)

    if train_base_model.upper() == 'Y':
        for layer in base_model.layers:
            layer.trainable = True
    elif train_base_model.upper() == 'N':
        for layer in base_model.layers:
            layer.trainable = False

    return model



def vgg19(train_base_model, dim, dense, dropout, output_layer, output_activation):
    print("\nTRAINING ON VGG19 MODEL:-")

    base_model = keras.applications.vgg19.VGG19(input_shape = dim, weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(dense)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    predictions = Dense(output_layer, activation = output_activation)(x)

    model = Model(inputs = base_model.input, outputs=predictions)

    if train_base_model.upper() == 'Y':
        for layer in base_model.layers:
            layer.trainable = True
    elif train_base_model.upper() == 'N':
        for layer in base_model.layers:
            layer.trainable = False

    return model



def MobileNet(train_base_model, dim, dense, dropout, output_layer, output_activation):
    print("\nTRAINING ON MobileNet MODEL:-")

    base_model = keras.applications.mobilenet.MobileNet(input_shape = dim, weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(dense)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    predictions = Dense(output_layer, activation = output_activation)(x)

    model = Model(inputs = base_model.input, outputs=predictions)

    if train_base_model.upper() == 'Y':
        for layer in base_model.layers:
            layer.trainable = True
    elif train_base_model.upper() == 'N':
        for layer in base_model.layers:
            layer.trainable = False

    return model



def InceptionV3(train_base_model, dim, dense, dropout, output_layer, output_activation):
    print("\nTRAINING ON InceptionV3 MODEL:-")
    
    base_model = keras.applications.inception_v3.InceptionV3(input_shape = dim, weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(dense)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    predictions = Dense(output_layer, activation = output_activation)(x)

    model = Model(inputs = base_model.input, outputs=predictions)

    if train_base_model.upper() == 'Y':
        for layer in base_model.layers:
            layer.trainable = True
    elif train_base_model.upper() == 'N':
        for layer in base_model.layers:
            layer.trainable = False

    return model



def ResNet50(train_base_model, dim, dense, dropout, output_layer, output_activation):
    print("\nTRAINING ON ResNet50 MODEL:-")

    base_model = keras.applications.resnet.ResNet50(input_shape = dim, weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(dense)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    predictions = Dense(output_layer, activation = output_activation)(x)

    model = Model(inputs = base_model.input, outputs=predictions)

    if train_base_model.upper() == 'Y':
        for layer in base_model.layers:
            layer.trainable = True
    elif train_base_model.upper() == 'N':
        for layer in base_model.layers:
            layer.trainable = False

    return model



def Custom_Model(dim, layer, conv_layer, conv, conv_size, dense, dropout, output_layer, output_activation):
    print("\nTRAINING ON A COMPLEX CUSTOM MODEL:-")

    model = keras.models.Sequential()
    for l in range(layer):
        l += 1
        m = (2**l)//2
        for c in range(conv_layer):
            model.add(Conv2D(conv*m, (conv_size, conv_size), padding = 'same', input_shape = dim))
            model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(dropout))
    model.add(Flatten())
    
    model.add(Dense(dense))
    model.add(LeakyReLU())
    model.add(Dropout(dropout))
    
    model.add(Dense(output_layer, activation=output_activation))
    return model



def Custom_Prebuilt_Model(dim, output_layer, output_activation):
    dim = dim
    conv = 16
    conv_size = 3
    layer = 5
    conv_layer = 1
    dense = 256
    dropout = 0.3
    
    print("\nTRAINING ON A CUSTOM PREBUILT MODEL:-")

    model = keras.models.Sequential()
    for l in range(layer):
        l += 1
        m = (2**l)//2
        for c in range(conv_layer):
            model.add(Conv2D(conv*m, (conv_size, conv_size), padding = 'same', input_shape = dim))
            model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(dropout))
    model.add(Flatten())
    
    model.add(Dense(dense))
    model.add(LeakyReLU())
    model.add(Dropout(dropout))
    
    model.add(Dense(output_layer, activation=output_activation))
    return model



def essentials(class_no):
    if class_no >= 2:
        print("This is a " + str(class_no) + "-Class Classification")
        output_activation = 'softmax'
        losses = 'categorical_crossentropy'
        class_mode = 'categorical'
        output_layer = class_no
    else:
        print("This is a Binary Classification")
        output_activation = 'sigmoid'
        losses = 'binary_crossentropy'
        class_mode = 'binary'
        output_layer = 1
        
    return output_activation, losses, class_mode, output_layer



def optimizer_selection(custom, optimizer_select, learning_rate):
    if custom.upper() == 'Y':
        if optimizer_select == 'SGD':
            optimizer = keras.optimizers.SGD(lr = learning_rate, decay = 1e-6, momentum = 0.9, nesterov = True)

        elif optimizer_select == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(learning_rate, rho = 0.9)

        elif optimizer_select == 'Adagrad':
            optimizer = keras.optimizers.Adagrad(learning_rate)

        elif optimizer_select == 'Adadelta':
            optimizer = keras.optimizers.Adadelta(learning_rate, rho = 0.95)

        elif optimizer_select == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)

        elif optimizer_select == 'Adamax':
            optimizer = keras.optimizers.Adamax(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999)

        elif optimizer_select == 'Nadam':
            optimizer = keras.optimizers.Nadam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999)
   
    elif custom.upper() == 'N':
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
    
    return optimizer


def generators(batch_size, class_mode, dim, train_directory, validation_directory, test_directory):
    train_datagen = ImageDataGenerator(rescale=1.0/255.)
    train_generator = train_datagen.flow_from_directory(train_directory,
                                                        batch_size = batch_size,
                                                        class_mode = class_mode,
                                                        target_size = (dim[0],dim[1]),
                                                        shuffle=True)

    validation_datagen = ImageDataGenerator(rescale=1.0/255.)
    validation_generator = validation_datagen.flow_from_directory(validation_directory,
                                                                  batch_size = batch_size,
                                                                  class_mode = class_mode,
                                                                  target_size = (dim[0],dim[1]),
                                                                  shuffle=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255.)
    test_generator = test_datagen.flow_from_directory(test_directory,
                                                      batch_size = batch_size,
                                                      class_mode = class_mode,
                                                      target_size = (dim[0],dim[1]),
                                                      shuffle=True)
    
    return train_generator, validation_generator, test_generator



def model_compile(model, optimizer, losses):
    model.compile(optimizer = optimizer, loss = losses, metrics = ['accuracy', 
                                                               tf.keras.metrics.Precision(), 
                                                               tf.keras.metrics.Recall()])

    model.summary()
    return model



def train(physical_devices, model, train_generator, validation_generator, test_generator, epoch, callback):
    if physical_devices != []:
        with tf.device("/GPU:0"):
            start = time.time()
            history = model.fit(train_generator,
                                epochs = epoch,
                                verbose = 1,
                                callbacks = callback,
                                validation_data = validation_generator,
                                shuffle=True)

            end = time.time()
            duration = end-start
    else:
        with tf.device("/CPU:0"):
            start = time.time()
            history = model.fit(train_generator,
                                epochs = epoch,
                                verbose = 1,
                                callbacks = callback,
                                validation_data = validation_generator,
                                shuffle=True)

            end = time.time()
            duration = end-start

    train_score = model.evaluate(train_generator)
    val_score = model.evaluate(validation_generator)
    test_score = model.evaluate(test_generator)
    
    return history, duration, train_score, val_score, test_score



def characteristics(history, char):
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))

    # TOTAL
    plt.plot(acc, 'r', label='Training Accuracy')
    plt.plot(val_acc, 'b', label='Validation Accuracy')
    plt.plot(loss, 'y', label='Training Loss')
    plt.plot(val_loss, 'g', label='Validation Loss')
    plt.xlabel("Epochs")
    plt.title('Training and Validation Characteristics')
    plt.legend()
    fig_name_jpg = "characteristics.jpg"
    fig_name_eps = "characteristics.eps"
    plt.savefig(os.path.join(char, fig_name_jpg))
    plt.savefig(os.path.join(char, fig_name_eps))
    plt.clf()
    plt.close()
    


def performance(train_score, val_score, test_score):
    training_accuracy = train_score[1]*100
    validation_accuracy = val_score[1]*100
    test_accuracy = test_score[1]*100
    test_precision = test_score[2]*100
    test_recall = test_score[3]*100
    
    return training_accuracy, validation_accuracy, test_accuracy, test_precision, test_recall



def pred(test_directory, test_generator, class_no, best_model_address, dim):
    test_class_list = []
    for test_name in os.listdir(test_directory):
        test = os.path.join(test_directory, test_name)
        test_class_list.append(test)
    test_class_list.sort()
    
    y_true = test_generator.classes
    labels = test_generator.class_indices
    
    y_pred = []
    tot = len(os.listdir(test_class_list[1]))*class_no

    best_model = load_model(best_model_address)

    
    with tqdm(total=tot) as pbar:
        for i in range(class_no):
            for filename in os.listdir(test_class_list[i]):
                file = os.path.join(test_class_list[i], filename)
                img = cv2.imread(file)
                res = cv2.resize(img, (dim[0], dim[1]))
                normed = res / 255.0
                im_arr = normed.reshape(1, dim[0], dim[1], dim[2])

                pred = best_model.predict(im_arr)
                pred_categorical = keras.utils.to_categorical(pred)

                if class_no >= 2:
                    max_pred = np.argmax(pred)
                else:
                    max_pred = np.argmax(pred_categorical)

                y_pred.append(max_pred)

                pbar.set_description("Progress")
                pbar.update()
                
    return y_true, y_pred, labels



def report(y_true, y_pred, labels):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix 
    
    print("Calculating CLASSIFICATION REPORT: ")
    classification_reports = classification_report(y_true, y_pred, target_names=labels)
    print(classification_reports)

    print("\nCalculating SENSITIVITY & SPECIFICITY..........:")
    cm = confusion_matrix(y_true, y_pred)
    total = sum(sum(cm))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("sensitivity = {:.4f}".format(sensitivity))
    print("specificity = {:.4f}".format(specificity))
    
    return cm, classification_reports, sensitivity, specificity



def conf_mat(cm, labels, char):
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=True,
                                class_names=labels,
                                show_normed=True)

    plt.savefig(os.path.join(char, 'confusion-matrix.eps'))
    plt.savefig(os.path.join(char, 'confusion-matrix.jpg'))



def save_readme(char,
                class_no,
                train_size,
                val_size,
                test_size,
                img_w,
                img_h,
                learning_rate,
                early_stop, 
                epoch,
                patience,
                req_epochs,
                batch_size,
                monitor,
                dropout,
                output_activation,
                losses,
                optimizer_select,
                tl_models,
                model,
                test_accuracy,
                test_precision,
                test_recall,
                classification_report,
                sensitivity,
                specificity,
                duration
                ):

    readme_name_text = "readme.txt"

    completeName_txt = os.path.join(char, readme_name_text) 

    readme = open(completeName_txt, "w")

    readme.write("This is a {}-Class Classification Task".format(int(class_no)))


    readme.write("\n\n\n--DATA SPLITTING--")
    readme.write("\n{} % data of the RAW Dataset is used for validation".format(train_size*100))
    readme.write("\n{} % data of the RAW Dataset is used for validation".format(val_size*100))
    readme.write("\n{} % data of the RAW Dataset is used for testing".format(test_size*100))


    readme.write("\n\n\n--DATA PRE-PROCESSINGS--")
    readme.write("\nThe images in the dataset have been rescaled to {}x{} pixels".format(img_w, img_h))


    readme.write("\n\n\n--HYPERPARAMETERS--")
    readme.write("\nInitial Learning Rate = {}".format(learning_rate))
    readme.write("\nMaximum No. of epochs = {}".format(epoch))
    readme.write("\nBatch Size = {}".format(batch_size))


    readme.write("\n\n\n--EARLY STOPPING--")
    if early_stop.upper() == 'Y':
        readme.write("\n{} with no improvement has been monitored for {} epochs".format(monitor, patience))
        readme.write("\nTraining stopped after monitoring {} for total {} epochs".format(monitor, req_epochs))
        readme.write("\nModel is saved where {} is properly optimized".format(monitor))
    elif early_stop.upper() == 'N':
        readme.write("\nModel is saved where {} is properly optimized".format(monitor))


    readme.write("\n\n\n--MODEL--")
    readme.write("\nActivation Function of hidden layers = LeakyReLU")
    readme.write("\nDropout = {}%".format(int(dropout*100)))
    readme.write("\nActivation function of the output layer = {}".format(output_activation))
    readme.write("\nCost function of the model = {}".format(losses))
    readme.write("\nOptimizer = {}\n\n".format(optimizer_select))

    if tl_models == 'Basic':
        readme.write("Trained on Basic Model\n")
        with redirect_stdout(readme):
            model.summary()

    if tl_models == 'VGG16':
        readme.write("Trained on VGG16\n")
        with redirect_stdout(readme):
            model.summary()

    if tl_models == 'VGG19':
        readme.write("Trained on VGG19\n")
        with redirect_stdout(readme):
            model.summary()

    if tl_models == 'MobileNet':
        readme.write("Trained on MobileNet\n")
        with redirect_stdout(readme):
            model.summary()

    if tl_models == 'Inception':
        readme.write("Trained on ImageNet\n")
        with redirect_stdout(readme):
            model.summary()

    if tl_models == 'ResNet50':
        readme.write("Trained on a ResNet Model\n")
        with redirect_stdout(readme):
            model.summary() 

    if tl_models == 'Own':
        readme.write("Trained on a Custom Model\n")
        with redirect_stdout(readme):
            model.summary()


    readme.write("\n\n--MODEL-PERFORMANCE--")
    readme.write("\nTest Accuracy = {}%".format(test_accuracy))
    readme.write("\nTest Precision = {}%".format(test_precision))
    readme.write("\nTest Recall = {}%".format(test_recall))

    readme.write("\n\n\n--CLASSIFICATION REPORT--\n")
    readme.write(classification_report)

    readme.write("\nSensitivity = {}%".format(sensitivity*100))
    readme.write("\nSpecificity = {}%".format(specificity*100))


    readme.write("\nExecution Time: {} seconds".format(duration))

    readme.write("\n\nCreated using Self-Regulated Image Classifier using Convolution Neural Network")

    readme.close()
    
