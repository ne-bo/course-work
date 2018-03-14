import os

import numpy as np
from PIL import Image
from tqdm import tqdm


class Character():
    def __init__(self, data_folder, subfolder, alphabet, character_number, character_representation_number):
        self.data_folder = data_folder
        self.subfolder = subfolder
        self.alphabet = alphabet
        self.character_number = character_number
        self.character_representation_number = character_representation_number

    def get_image(self):
        image_path = self.data_folder + '/' + self.subfolder \
                     + '/' + self.alphabet + '/' + self.character_number \
                     + '/' + self.character_representation_number
        #print('image_path', image_path)
        self.image = Image.open(image_path)
        #plt.imshow(np.asarray(self.image))
        # plt.show()
        return self.image

    def get_label(self):
        self.label = self.alphabet + self.character_number
        return self.label


def get_list_of_all_possible_characters():
    all_characters = []
    data_folder = 'omniglot'
    subfolder = 'images_background'
    alphabetes = os.listdir(data_folder + '/' + subfolder)
    print('alphabetes', alphabetes)
    for alphabet in alphabetes:
        characters_numbers = os.listdir(data_folder + '/' + subfolder + '/' + alphabet)
        for character_number in characters_numbers:
            character_representations_numbers = os.listdir(data_folder + '/' + subfolder + '/' +
                                                           alphabet + '/' + character_number)
            for character_representation_number in character_representations_numbers:
                new_character = Character(data_folder, subfolder, alphabet,
                                          character_number, character_representation_number)
                all_characters.append(new_character)
    return np.array(all_characters)


def get_list_of_images_of_selected_characters(indices_of_selected_characters, all_characters):
    selected_characters = all_characters[indices_of_selected_characters]
    list_of_images_of_selected_characters = []
    list_of_labels_of_selected_characters = []
    for character in selected_characters:
        list_of_images_of_selected_characters.append(character.get_image())
        list_of_labels_of_selected_characters.append(character.get_label())
    return list_of_images_of_selected_characters, list_of_labels_of_selected_characters


def create_a_new_image_with_selected_characters(list_of_images_of_selected_characters):
    small_image_sizex, small_image_sizey = list_of_images_of_selected_characters[0].size
    big_image_size = int(small_image_sizex * 3.0 * np.sqrt(2.0))
    positions_size = int(big_image_size * 2 / 3)
    new_image = Image.new("RGB", (big_image_size, big_image_size), (255, 255, 255))
    for image in list_of_images_of_selected_characters:
        rgba_image = image.convert('RGBA')
        # rotated image
        rot = rgba_image.rotate(np.random.randint(low=0, high=361), expand=1)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,)*4)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)
        randomized_small_image_position_x, \
        randomized_small_image_position_y = \
            np.random.randint(low=0, high=positions_size), np.random.randint(low=0, high=positions_size)
        new_image.paste(out, (randomized_small_image_position_x, randomized_small_image_position_y))
    new_image = new_image.resize((105, 105))
    return new_image


def generate_one_image(maximum_number_of_characters_on_image, list_of_all_possible_characters):
    how_many_characters_we_want_to_have = np.random.randint(low=1, high=maximum_number_of_characters_on_image + 1)
    how_many_characters_we_have = len(list_of_all_possible_characters)
    indices_of_selected_characters = []
    for i in range(how_many_characters_we_want_to_have):
        index_of_selected_character = np.random.randint(low=0, high=how_many_characters_we_have)
        indices_of_selected_characters.append(index_of_selected_character)
    #print('indices_of_selected_characters ', indices_of_selected_characters)


    list_of_images_of_selected_characters, \
    list_of_labels_of_selected_characters = get_list_of_images_of_selected_characters(indices_of_selected_characters,
                                                                                      list_of_all_possible_characters)
    new_image = create_a_new_image_with_selected_characters(list_of_images_of_selected_characters)
    return new_image, list_of_labels_of_selected_characters


def generate_centaurus_omniglot_dataset(total_number_of_images, maximum_number_of_characters_on_image):
    list_of_all_possible_characters = get_list_of_all_possible_characters()
    with open('labels_natasha_omniglot_train', "w") as fout:
        for i in tqdm(range(total_number_of_images)):
            image, list_of_labels = generate_one_image(maximum_number_of_characters_on_image,
                                                   list_of_all_possible_characters)
            image.convert('RGB').save('natasha_omniglot_train/%d.jpg' %i)
            fout.write('%d'% i)
            for label in list_of_labels:
                fout.write(' ' + label)
            fout.write('\n')


generate_centaurus_omniglot_dataset(total_number_of_images=100000, maximum_number_of_characters_on_image=3)