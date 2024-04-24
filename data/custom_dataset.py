import json 
from time import time 
from utils import language_utils
import functools
print = functools.partial(print, flush=True)

class CustomDataset:
    TrainSet_ID = 1
    ValidationSet_ID = 2
    TestSet_ID = 3

    def __init__(self,
                images_path,
                annotations_path,
                precalc_features_hdf5_filepath,
                preproc_images_hdf5_filepath=None,
                dict_min_occurrences=1,
                verbose=True
                ):
        super(CustomDataset, self).__init__()

        self.use_images_instead_of_features = False
        if precalc_features_hdf5_filepath is None or precalc_features_hdf5_filepath == 'None' or \
                precalc_features_hdf5_filepath == 'none' or precalc_features_hdf5_filepath == '':
            self.use_images_instead_of_features = True
            print("Warning: since no hdf5 path is provided using images instead of pre-calculated features.")
            print("Features path: " + str(precalc_features_hdf5_filepath))

            self.preproc_images_hdf5_filepath = None
            if preproc_images_hdf5_filepath is not None:
                print("Preprocessed hdf5 file path not None: " + str(preproc_images_hdf5_filepath))
                print("Using preprocessed hdf5 file instead.")
                self.preproc_images_hdf5_filepath = preproc_images_hdf5_filepath

        else:
            self.precalc_features_hdf5_filepath = precalc_features_hdf5_filepath
            print("Features path: " + str(self.precalc_features_hdf5_filepath))
            print("Features path provided, images are provided in form of features.")

        if images_path is None:
            self.images_path = ""
        else:
            self.images_path = images_path

        self.train_list = []
        self.val_list = []
        self.test_list = []

        #   YOUR CUSTOM DATA SET DEFINITION
        # definition of an item
        # item is a dictionary with the following elements
        # {'img_path': where the image is located
        # 'img_id': unique id amnong all test + train + val splits, make sure to use different ids
        # 'captions': list of strings, note that even if you have just one caption, it is expected a
        #                              list of strings anyway
        # }

        with open(annotations_path, 'r') as f:
            json_file = json.load(f)

        for json_item in json_file:
            new_item = dict()

            new_item['img_path'] = self.images_path + json_item['path']
            new_item['img_id'] = json_item['img_id']
            new_item['captions'] = json_item['captions']
            print(new_item)

            if json_item['split'] == 'train':
                self.train_list.append(new_item)
            elif json_item['split'] == 'test':
                self.test_list.append(new_item)
            elif json_item['split'] == 'val':
                self.val_list.append(new_item)

        ## END CUSTOM PART
        
        self.train_num_images = len(self.train_list)
        self.val_num_images = len(self.val_list)
        self.test_num_images = len(self.test_list)

        if verbose:
            print("Num train images: " + str(self.train_num_images))
            print("Num val images: " + str(self.val_num_images))
            print("Num test images: " + str(self.test_num_images))

        tokenized_captions_list = []
        for i in range(self.train_num_images):
            for caption in self.train_list[i]['captions']:
                tmp = language_utils.lowercase_and_clean_trailing_spaces([caption])
                tmp = language_utils.add_space_between_non_alphanumeric_symbols(tmp)
                tmp = language_utils.remove_punctuations(tmp)
                tokenized_caption = ['SOS'] + language_utils.tokenize(tmp)[0] + ['EOS']
                tokenized_captions_list.append(tokenized_caption)

        counter_dict = dict()
        for i in range(len(tokenized_captions_list)):
            for word in tokenized_captions_list[i]:
                if word not in counter_dict:
                    counter_dict[word] = 1
                else:
                    counter_dict[word] += 1

        less_than_min_occurrences_set = set()
        for k, v in counter_dict.items():
            if v < dict_min_occurrences:
                less_than_min_occurrences_set.add(k)
        if verbose:
            print("tot tokens " + str(len(counter_dict)) +
                " less than " + str(dict_min_occurrences) + ": " + str(len(less_than_min_occurrences_set)) +
                " remaining: " + str(len(counter_dict) - len(less_than_min_occurrences_set)))

        self.num_caption_vocab = 4
        self.max_seq_len = 0
        discovered_words = ['PAD', 'SOS', 'EOS', 'UNK']
        for i in range(len(tokenized_captions_list)):
            caption = tokenized_captions_list[i]
            if len(caption) > self.max_seq_len:
                self.max_seq_len = len(caption)
            for word in caption:
                if (word not in discovered_words) and (not word in less_than_min_occurrences_set):
                    discovered_words.append(word)
                    self.num_caption_vocab += 1

        discovered_words.sort()
        self.caption_word2idx_dict = dict()
        self.caption_idx2word_list = []
        for i in range(len(discovered_words)):
            self.caption_word2idx_dict[discovered_words[i]] = i
            self.caption_idx2word_list.append(discovered_words[i])
        if verbose:
            print("There are " + str(self.num_caption_vocab) + " vocabs in dict")

    def get_image_path(self, img_idx, dataset_split):

        if dataset_split == CustomDataset.TestSet_ID:
            img_path = self.test_list[img_idx]['img_path']
            img_id = self.test_list[img_idx]['img_id']
        elif dataset_split == CustomDataset.ValidationSet_ID:
            img_path = self.val_list[img_idx]['img_path']
            img_id = self.val_list[img_idx]['img_id']
        else:
            img_path = self.train_list[img_idx]['img_path']
            img_id = self.train_list[img_idx]['img_id']

        return img_path, img_id

    def get_all_images_captions(self, dataset_split):
        all_image_references = []

        if dataset_split == CustomDataset.TestSet_ID:
            dataset = self.test_list
        elif dataset_split == CustomDataset.ValidationSet_ID:
            dataset = self.val_list
        else:
            dataset = self.train_list

        for img_idx in range(len(dataset)):
            all_image_references.append(dataset[img_idx]['captions'])
        return all_image_references

    def get_eos_token_idx(self):
        return self.caption_word2idx_dict['EOS']

    def get_sos_token_idx(self):
        return self.caption_word2idx_dict['SOS']

    def get_pad_token_idx(self):
        return self.caption_word2idx_dict['PAD']

    def get_unk_token_idx(self):
        return self.caption_word2idx_dict['UNK']

    def get_eos_token_str(self):
        return 'EOS'

    def get_sos_token_str(self):
        return 'SOS'

    def get_pad_token_str(self):
        return 'PAD'

    def get_unk_token_str(self):
        return 'UNK'