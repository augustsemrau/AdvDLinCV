import os
from PIL import Image
from torch.utils.data.dataset import Dataset


class TripletDataset(Dataset):
    """
    Each triplet has its folder with the corresponding 3 images.
    Structure of the triplet dataset:
        triplet_<number>
            -   1.png
            -   2.png
            -   3.png
    """
    def __init__(self, triplets_path, transform):
        self.triplets_path = triplets_path
        self.transform = transform
        self.triplets = list(os.listdir(triplets_path))


    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, index):
        current_path = self.triplets[index] # e.g. triplet triplet_000010
        
        # Make sure you understand the __init__ function and the structure of the data first.

        # TASK 1: Read the triplet and make sure you use the self.transform
        frame_1 = os.path.join(self.triplets_path, current_path, "1.png")
        frame_2 = os.path.join(self.triplets_path, current_path, "2.png")
        frame_3 = os.path.join(self.triplets_path, current_path, "3.png")

        frame_1 = Image.open(frame_1)
        frame_2 = Image.open(frame_2)
        frame_3 = Image.open(frame_3)

        frame_1 = self.transform(frame_1)
        frame_2 = self.transform(frame_2)
        frame_3 = self.transform(frame_3)

        return frame_1, frame_2, frame_3
