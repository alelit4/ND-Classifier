''' Alexandra Rivero 
   :D  '''

import argparse 
import utility_functions as utilities
import model_functions as md
from torchvision import datasets, transforms, models
from collections import OrderedDict

DEBUG = True


default_path_to_image = "./flowers/test/10/image_07117.jpg"
def init_parser():
    my_argparse = argparse.ArgumentParser()
    my_argparse.add_argument('--path_to_image', nargs='*', type = str,  help='Path to path_to_image', default=default_path_to_image)
    my_argparse.add_argument('checkpoint', nargs='*', type = str,  help='Path to checkpoint', default="my_checkpoint.pth")
    my_argparse.add_argument('--top_k', '--k', default=5, type=int, help='Number of Top K selection (in predictions)')
    my_argparse.add_argument('--gpu', action='store_true', help='To activate gpu mode', default=False)
    my_argparse.add_argument('--category_names', '--cn', type=str, help='File to load the category names', default="cat_to_name.json")

    return my_argparse.parse_args()


def main():
    print("Let's go!")
    custom_args = init_parser()
    device = utilities.get_device(custom_args.gpu)
    print("=>> --gpu '{}'".format(custom_args.gpu))
    if DEBUG:
        print("Loading categories_to_names")
    categories_to_names = utilities.load_cat_to_names(custom_args.category_names)
    if DEBUG:
        print("Loading my checkpoint")
    my_new_model, my_new_optimizer, my_new_criterion = utilities.load_my_checkpoint(custom_args.checkpoint)
    if DEBUG:
        print("Doing the prediction")
    test_image = utilities.process_image(custom_args.path_to_image)
    probs, classes = md.predict(custom_args.path_to_image, my_new_model, custom_args.top_k, device)
    if DEBUG:
        print("Printing the prediction")
    print("probs => {}".format(probs))
    print("classes => {}".format(classes))

    for i in range(len(classes)):
        print("%d - This image is a %s with an accuracy of %.2f %%" % (i + 1, categories_to_names.get(classes[i]).capitalize(), probs[i] * 100))
    #utilities.print_results(custom_args.path_to_image)
    
if __name__ == '__main__':
    main()