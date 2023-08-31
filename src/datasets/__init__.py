from datasets.CityscapesDataset import CityscapesDataset
from datasets.RobomisDataset import Robomis

def get_dataset(name, dataset_opts):
    if name == "cityscapes": 
        return CityscapesDataset(**dataset_opts)
    if name == "robomis":
        return Robomis(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))