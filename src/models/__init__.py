from models.BranchedERFNet import BranchedERFNet
from models.unet import BranchedORUnet

def get_model(name, model_opts):
    if name == "branched_erfnet":
        model = BranchedERFNet(**model_opts)
        return model
    if name == "branched_orunet":
        model = BranchedORUnet(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))