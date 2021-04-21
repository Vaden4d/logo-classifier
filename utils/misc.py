import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def predict_on_loader(loader, model, device):
    """Make prediction over custom loader
    Used to obtain predictions on separate
    validaiton data (not part of training pipeline)
    """
    model.eval()
    model = model.to(device)
    
    pred = []
    with tqdm(ascii=True, leave=False, total=len(loader)) as bar:
        for batch in loader:
            
            images, reals = batch["features"], batch["targets"]
            with torch.no_grad():

                preds = model(images.to(device))
                preds = F.softmax(preds, dim=-1).cpu()

            pred += preds[:, 1].tolist()

            bar.update()

    return pred
