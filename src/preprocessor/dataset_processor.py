import numpy as np
# def generate_synthetic_images(img, save_dir, samples=20, save_prefix=""):
#     datagen = ImageDataGenerator(
#             rotation_range=40,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             shear_range=0.2,
#             zoom_range=0.2,
#             horizontal_flip=True,
#             fill_mode='nearest')

#     #img = load_img(img_file)  # this is a PIL image
#     x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#     x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#     # the .flow() command below generates batches of randomly transformed images
#     # and saves the results to the `preview/` directory
#     i = 0
#     for batch in datagen.flow(x, batch_size=1,
#                             save_to_dir=save_dir, save_prefix=save_prefix, save_format='jpeg'):
#         i += 1
#         if i >= samples:
#             break  # otherwise the generator would loop indefinitely

def split_data(data_set):
    x = np.asarray(list(map(lambda x: data_set[x][0], data_set.keys())))
    y = np.asarray(list(map(lambda x: data_set[x][1], data_set.keys())))
    return x, y

if __name__ == "__main__":
    from helpers.constants import Path
    from image_processor import load_image, prepare_image, feature_extract_image, load_model
    import os
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

    from helpers.io import save_pickle
    root_path = os.path.join(Path.img_original)
    img_dirs = list(filter(lambda x: ".DS" not in x, os.listdir(root_path)))
    num_classes = len(img_dirs)
    
    train_set = dict()
    val_set = dict()
    test_set = dict()

    num_train_sample = 0
    num_val_sample = 0
    num_test_sample = 0
    # load pretrained model here!
    model = load_model()
    for class_index, img_dir in enumerate(img_dirs):
        
        path_images = os.path.join(root_path, img_dir)
        image_files = list(filter(lambda x: ".DS" not in x, os.listdir(path_images)))
        num_images = len(image_files)
        # index splitting in train, test and val
        first_index = int(num_images * 0.8)
        second_index = first_index + ((num_images - first_index) // 2)
        
        for data_split_index, img_file in enumerate(image_files):
            img_file = os.path.join(path_images, img_file)
            
            # rezise image to 299 and pad with black
            img = load_image(img_file)
            img = prepare_image(img, 299)
            
            # create onehot vector
            one_hot = np.zeros(num_classes)
            one_hot[class_index] = 1

            feature = feature_extract_image(img, model)
            if data_split_index < first_index:
                num_train_sample += 1
                train_set["sample_" + str(num_train_sample)] = np.asarray([feature, one_hot])
                
            elif data_split_index < second_index:
                num_val_sample += 1
                val_set["sample_" + str(num_val_sample)] = np.asarray([feature, one_hot])
            else:
                num_test_sample += 1
                test_set["sample_" + str(num_test_sample)] = np.asarray([feature, one_hot])

            if data_split_index % 10 == 0:
                print("class: ", class_index, "Images extracted: ", data_split_index)
      
    save_pickle(Path.train,"train.pickle", train_set)
    save_pickle(Path.validate, "validate.pickle", val_set)
    save_pickle(Path.test, "test.pickle",test_set)
            


    