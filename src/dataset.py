import torch
from torch._prims_common import mask_tensor
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np 


# Dictionary mapping original food items to synonyms/equivalent items
# for zero-shot segmentation testing

food_synonyms = {
    0: "background",
    1: "sweets",  # candy
    2: "custard tart",  # egg tart
    3: "chips",  # french fries
    4: "cocoa",  # chocolate
    5: "cookie",  # biscuit
    6: "popped corn",  # popcorn
    7: "custard",  # pudding
    8: "gelato",  # ice cream
    9: "butter cheese",  # cheese butter
    10: "pastry",  # cake
    11: "red wine",  # wine
    12: "shake",  # milkshake
    13: "espresso",  # coffee
    14: "fruit juice",  # juice
    15: "dairy milk",  # milk
    16: "green tea",  # tea
    17: "almond nut",  # almond
    18: "kidney beans",  # red beans
    19: "cashew nut",  # cashew
    20: "dried berries",  # dried cranberries
    21: "soybean",  # soy
    22: "walnut nut",  # walnut
    23: "groundnut",  # peanut
    24: "chicken egg",  # egg
    25: "red apple",  # apple
    26: "date fruit",  # date
    27: "dried apricot",  # apricot
    28: "avocado pear",  # avocado
    29: "yellow banana",  # banana
    30: "fresh strawberry",  # strawberry
    31: "cherry fruit",  # cherry
    32: "fresh blueberry",  # blueberry
    33: "red raspberry",  # raspberry
    34: "tropical mango",  # mango
    35: "olive fruit",  # olives
    36: "peach fruit",  # peach
    37: "yellow lemon",  # lemon
    38: "pear fruit",  # pear
    39: "fig fruit",  # fig
    40: "tropical pineapple",  # pineapple
    41: "table grape",  # grape
    42: "kiwi fruit",  # kiwi
    43: "cantaloupe",  # melon
    44: "mandarin orange",  # orange
    45: "summer watermelon",  # watermelon
    46: "beef steak",  # steak
    47: "pork meat",  # pork
    48: "poultry",  # chicken duck
    49: "hot dog",  # sausage
    50: "fried chicken",  # fried meat
    51: "mutton",  # lamb
    52: "gravy",  # sauce
    53: "crab meat",  # crab
    54: "seafood fish",  # fish
    55: "clam",  # shellfish
    56: "prawn",  # shrimp
    57: "broth",  # soup
    58: "loaf",  # bread
    59: "maize",  # corn
    60: "burger",  # hamburg
    61: "flatbread pizza",  # pizza
    62: "steamed buns",  # hanamaki baozi
    63: "dumplings",  # wonton dumplings
    64: "spaghetti",  # pasta
    65: "ramen",  # noodles
    66: "steamed rice",  # rice
    67: "tart",  # pie
    68: "bean curd",  # tofu
    69: "aubergine",  # eggplant
    70: "spud",  # potato
    71: "garlic clove",  # garlic
    72: "white cauliflower",  # cauliflower
    73: "red tomato",  # tomato
    74: "sea kelp",  # kelp
    75: "nori",  # seaweed
    76: "scallion",  # spring onion
    77: "canola",  # rape
    78: "ginger root",  # ginger
    79: "lady finger",  # okra
    80: "green lettuce",  # lettuce
    81: "squash",  # pumpkin
    82: "green cucumber",  # cucumber
    83: "daikon",  # white radish
    84: "orange carrot",  # carrot
    85: "asparagus spear",  # asparagus
    86: "bamboo shoot",  # bamboo shoots
    87: "green broccoli",  # broccoli
    88: "celery stalk",  # celery stick
    89: "coriander",  # cilantro mint
    90: "sugar snap peas",  # snow peas
    91: "green cabbage",  # cabbage
    92: "sprouts",  # bean sprouts
    93: "yellow onion",  # onion
    94: "bell pepper",  # pepper
    95: "string beans",  # green beans
    96: "haricot verts",  # French beans
    97: "king trumpet mushroom",  # king oyster mushroom
    98: "shiitake mushroom",  # shiitake
    99: "enoki",  # enoki mushroom
    100: "oyster fungi",  # oyster mushroom
    101: "champignon",  # white button mushroom
    102: "mixed greens",  # salad
    103: "miscellaneous ingredients"  # other ingredients
}



class FoodSeg103Dataset(Dataset):
    def __init__(self, root,split='train', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform  
        self.image_dir = os.path.join(root, "Images", "img_dir",split)
        self.mask_dir = os.path.join(root, "Images", "ann_dir",split)
        self.split=split



        
        with open(os.path.join(root, "ImageSets", f"{split}.txt"), "r") as f:
            ids = [line.strip() for line in f.readlines()]

        
        self.images = [os.path.join(self.image_dir, img_id) for img_id in ids]
        self.masks = [os.path.join(self.mask_dir, img_id.split('.')[0] + '.png') for img_id in ids]

        
        self.labels = self._load_labels()

    def _load_labels(self):
        """Reads category_id.txt and returns a list of class names."""
        labels = []
        with open(os.path.join(self.root, "category_id.txt"), "r") as f:
            for line in f:
                parts = line.strip().split()  # split on any whitespace
                if len(parts) < 2:
                    continue  # skip empty/malformed lines
                name_string = " ".join(parts[1:])  # join all words after ID
                labels.append(name_string)
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        

        return image, mask
