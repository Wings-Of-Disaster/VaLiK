import os
import numpy as np
from termcolor import cprint
from transformers import BertTokenizer
import torch.utils.data as data

from PIL import Image

task_dict = {
    'task1': 'informative',
    'task2': 'humanitarian',
    'task2_merged': 'humanitarian',
}

labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}

labels_task2_merged = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)


def expand2square(pil_img, background_color=(0, 0, 0)):
    # Reference: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class CrisisMMDataset(BaseDataset):

    def read_data(self, ann_file):
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            event_name, tweet_id, image_id, tweet_text, image, label, label_text, label_image, label_text_image = l.split(
                '\t')

            if self.consistent_only and label_text != label_image:
                continue

            combined_text = f"{tweet_text}"
            self.data_list.append(combined_text)

    def initialize(self, phase='train', cat='all', task='task2', shuffle=False, consistent_only=False):
        self.dataset_root = '../../datasets/CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.label_map = None
        self.consistent_only = False
        if task == 'task1':
            self.label_map = labels_task1
        elif task == 'task2':
            self.label_map = labels_task2
        elif task == 'task2_merged':
            self.label_map = labels_task2_merged

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        ann_file = '%s/crisismmd_datasplit_all/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        )

        self.read_data(ann_file)

        if shuffle:
            np.random.default_rng(seed=0).shuffle(self.data_list)
        self.data_list = self.data_list[:2147483648]
        cprint('[*] %d samples loaded.' % (len(self.data_list)), 'yellow')

        self.N = len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'CrisisMMDataset'


if __name__ == "__main__":

    dataset = CrisisMMDataset()
    dataset.initialize(phase='train', task='task2')
    all_texts = [dataset[i] for i in range(len(dataset))]
    all_texts = list(set(all_texts))

    combined_all_texts = " ".join(all_texts)
    with open("CrisisMMD_Text.txt", "w", encoding="utf-8") as f:
        f.write(combined_all_texts)