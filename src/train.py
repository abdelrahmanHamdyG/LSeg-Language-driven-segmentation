# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import tempfile
from src.models.lseg_model import LSegModel
from src.dataset import FoodSeg103Dataset
from src.utils.metrics import  intersection_and_union
from src.utils.loss import SegmentationLoss
import torchvision.transforms.functional as TF

# ----------------------------
# Config / hyperparams
# ----------------------------

food_synonyms = [
    "background",  # 0
    "sweets",  # 1 - candy
    "custard tart",  # 2 - egg tart
    "chips",  # 3 - french fries
    "cocoa",  # 4 - chocolate
    "cookie",  # 5 - biscuit
    "popped corn",  # 6 - popcorn
    "custard",  # 7 - pudding
    "gelato",  # 8 - ice cream
    "butter cheese",  # 9 - cheese butter
    "pastry",  # 10 - cake
    "red wine",  # 11 - wine
    "shake",  # 12 - milkshake
    "espresso",  # 13 - coffee
    "fruit juice",  # 14 - juice
    "dairy milk",  # 15 - milk
    "green tea",  # 16 - tea
    "almond nut",  # 17 - almond
    "kidney beans",  # 18 - red beans
    "cashew nut",  # 19 - cashew
    "dried berries",  # 20 - dried cranberries
    "soybean",  # 21 - soy
    "walnut nut",  # 22 - walnut
    "groundnut",  # 23 - peanut
    "chicken egg",  # 24 - egg
    "red apple",  # 25 - apple
    "date fruit",  # 26 - date
    "dried apricot",  # 27 - apricot
    "avocado pear",  # 28 - avocado
    "yellow banana",  # 29 - banana
    "fresh strawberry",  # 30 - strawberry
    "cherry fruit",  # 31 - cherry
    "fresh blueberry",  # 32 - blueberry
    "red raspberry",  # 33 - raspberry
    "tropical mango",  # 34 - mango
    "olive fruit",  # 35 - olives
    "peach fruit",  # 36 - peach
    "yellow lemon",  # 37 - lemon
    "pear fruit",  # 38 - pear
    "fig fruit",  # 39 - fig
    "tropical pineapple",  # 40 - pineapple
    "table grape",  # 41 - grape
    "kiwi fruit",  # 42 - kiwi
    "cantaloupe",  # 43 - melon
    "mandarin orange",  # 44 - orange
    "summer watermelon",  # 45 - watermelon
    "beef steak",  # 46 - steak
    "pork meat",  # 47 - pork
    "poultry",  # 48 - chicken duck
    "hot dog",  # 49 - sausage
    "fried chicken",  # 50 - fried meat
    "mutton",  # 51 - lamb
    "gravy",  # 52 - sauce
    "crab meat",  # 53 - crab
    "seafood fish",  # 54 - fish
    "clam",  # 55 - shellfish
    "prawn",  # 56 - shrimp
    "broth",  # 57 - soup
    "loaf",  # 58 - bread
    "maize",  # 59 - corn
    "burger",  # 60 - hamburg
    "flatbread pizza",  # 61 - pizza
    "steamed buns",  # 62 - hanamaki baozi
    "dumplings",  # 63 - wonton dumplings
    "spaghetti",  # 64 - pasta
    "ramen",  # 65 - noodles
    "steamed rice",  # 66 - rice
    "tart",  # 67 - pie
    "bean curd",  # 68 - tofu
    "aubergine",  # 69 - eggplant
    "spud",  # 70 - potato
    "garlic clove",  # 71 - garlic
    "white cauliflower",  # 72 - cauliflower
    "red tomato",  # 73 - tomato
    "sea kelp",  # 74 - kelp
    "nori",  # 75 - seaweed
    "scallion",  # 76 - spring onion
    "canola",  # 77 - rape
    "ginger root",  # 78 - ginger
    "lady finger",  # 79 - okra
    "green lettuce",  # 80 - lettuce
    "squash",  # 81 - pumpkin
    "green cucumber",  # 82 - cucumber
    "daikon",  # 83 - white radish
    "orange carrot",  # 84 - carrot
    "asparagus spear",  # 85 - asparagus
    "bamboo shoot",  # 86 - bamboo shoots
    "green broccoli",  # 87 - broccoli
    "celery stalk",  # 88 - celery stick
    "coriander",  # 89 - cilantro mint
    "sugar snap peas",  # 90 - snow peas
    "green cabbage",  # 91 - cabbage
    "sprouts",  # 92 - bean sprouts
    "yellow onion",  # 93 - onion
    "bell pepper",  # 94 - pepper
    "string beans",  # 95 - green beans
    "haricot verts",  # 96 - French beans
    "king trumpet mushroom",  # 97 - king oyster mushroom
    "shiitake mushroom",  # 98 - shiitake
    "enoki",  # 99 - enoki mushroom
    "oyster fungi",  # 100 - oyster mushroom
    "champignon",  # 101 - white button mushroom
    "mixed greens",  # 102 - salad
    "miscellaneous ingredients"  # 103 - other ingredients
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_FROM = "checkpoints/lseg_epoch36.pt"  # set to checkpoint path if you want to resume
IMG_SIZE = 224           
BATCH_SIZE = 16
NUM_EPOCHS = 40         
SAVE_EVERY = 3
BACKBONE_UNFREEZE_EPOCH = 3   

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# learning rates
LR_HEAD = 1e-3          # head/decoder lr
LR_BACKBONE = 1e-5      # small backbone lr after unfreeze
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0        # set 0 on Windows if you had issues
device = DEVICE

# ----------------------------
# Prompt templates (training)
# ----------------------------
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a picture of {}",
    "a close-up of {}",
    "a plate of {}",
    "a dish containing {}",
]


image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

def mask_transform(mask: Image.Image):
    mask = mask.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    mask = torch.from_numpy(np.array(mask)).long()
    return mask

def evaluate_zero_shot(model, dataset_split='test'):
    print(f"Starting zero-shot evaluation on {dataset_split} split...")
    model.eval()

    num_classes = 104  # total classes
    total_inter = torch.zeros(num_classes).to(device)
    total_union = torch.zeros(num_classes).to(device)

    # Precompute all text embeddings once (all 104 classes)
    with torch.no_grad():
        all_text_features = build_prompt_features(model, food_synonyms, PROMPT_TEMPLATES).to(device)

    # Load dataset
    eval_dataset = FoodSeg103Dataset(
        root="data/FoodSeg103",
        split=dataset_split,
        transform=image_transform,
        target_transform=mask_transform,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Iterate over dataset
    for i, (image, mask) in enumerate(eval_loader):
        image = image.to(device)
        mask = mask.to(device).squeeze(0)  # remove batch dim

        with torch.no_grad():
            outputs = model(image, text_features=all_text_features)  # [1, num_classes, H, W]
            preds = torch.argmax(outputs, dim=1).squeeze(0)        # [H, W]

    
        inter, union = intersection_and_union(preds, mask, num_classes)
        total_inter += inter
        total_union += union

        if (i + 1) % 500 == 0:
            miou_so_far = (total_inter / (total_union + 1e-10)).mean().item()
            print(f"[Eval {i+1}/{len(eval_loader)}] mIoU so far: {miou_so_far:.4f}")

    # Final mIoU
    iou = total_inter / (total_union + 1e-10)
    miou = iou.mean().item()
    print("Evaluation complete.")
    print(f"Final mIoU: {miou:.4f}")

    return miou

def build_prompt_features(model, categories, templates):
    """
    Encode multiple templates per category, average them, normalize, return [N, 512] tensor on DEVICE.
    model.text_encoder expects a python list of strings and returns a [T, 512] tensor.
    """
    all_embeddings = []
    model.eval()  
    with torch.no_grad():
        for cat in categories:
            prompts = [t.format(cat) for t in templates]
            feats = model.text_encoder(prompts).to(DEVICE).float()  # [T, C]
            mean_feat = feats.mean(dim=0, keepdim=True)  # [1, C]
            all_embeddings.append(mean_feat)
    text_features = torch.cat(all_embeddings, dim=0)  # [N, C]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  

# ----------------------------

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, miou=None, scaler=None, keep_last_k=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt", prefix="tmp_ckpt_", dir=os.path.dirname(path))
    os.close(tmp_fd)

    state = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "miou": miou,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()

    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    if keep_last_k is not None and keep_last_k > 0:
        ckpt_dir = os.path.dirname(path)
        files = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        to_delete = files[:-keep_last_k]
        for f in to_delete:
            try:
                os.remove(f)
            except Exception:
                pass

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device=torch.device("cpu"), strict_model=True):
    if path is None or not os.path.exists(path):
        return 0, 0.0

    checkpoint = torch.load(path, map_location=device)
    model_state = checkpoint.get("model_state", None)
    if model_state is None:
        raise KeyError("Checkpoint does not contain 'model_state'")

    if hasattr(model, "module") and not any(k.startswith("module.") for k in model_state.keys()):
        new_state = {}
        for k, v in model_state.items():
            new_state["module." + k] = v
        model_state = new_state
    elif not hasattr(model, "module") and any(k.startswith("module.") for k in model_state.keys()):
        new_state = {}
        for k, v in model_state.items():
            new_state[k.replace("module.", "", 1)] = v
        model_state = new_state

    model.load_state_dict(model_state, strict=strict_model)
    print(f"Loaded model weights from {path}")

    start_epoch = checkpoint.get("epoch", 0)
    best_miou = checkpoint.get("miou", 0.0)

    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("Loaded optimizer state.")
        except Exception as e:
            print(f"Warning: optimizer state not loaded ({e}). You may reinitialize optimizer.")
    if scheduler is not None and "scheduler_state" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("Loaded scheduler state.")
        except Exception as e:
            print(f"Warning: scheduler state not loaded ({e}).")
    if scaler is not None and "scaler_state" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state"])
            print("Loaded amp scaler state.")
        except Exception as e:
            print(f"Warning: amp scaler not loaded ({e}).")

    return int(start_epoch), float(best_miou)

# ----------------------------
# LR scheduler
# ----------------------------
from torch.optim.lr_scheduler import _LRScheduler
class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, power=0.9, last_epoch=-1, verbose=False):
        self.total_iters = max(1, int(total_iters))
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        it = min(self.last_epoch, self.total_iters)
        factor = (1 - it / float(self.total_iters)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

def _no_weight_decay(name: str):
    name_l = name.lower()
    if name_l.endswith(".bias"):
        return True
    if "bn" in name_l or "layernorm" in name_l or "ln" in name_l or "norm" in name_l:
        return True
    if "bias" in name_l or "absolute_pos_embed" in name_l or "relative_position" in name_l:
        return True
    return False

def build_optimizer(model_obj):
    head_params_wd = []
    head_params_no_wd = []
    backbone_params_wd = []
    backbone_params_no_wd = []

    for name, p in model_obj.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("dpt_backbone"):
            if _no_weight_decay(name):
                backbone_params_no_wd.append(p)
            else:
                backbone_params_wd.append(p)
        else:
            if _no_weight_decay(name):
                head_params_no_wd.append(p)
            else:
                head_params_wd.append(p)

    param_groups = []
    if head_params_wd:
        param_groups.append({"params": head_params_wd, "lr": LR_HEAD, "weight_decay": 0.01, "name": "head_wd"})
    if head_params_no_wd:
        param_groups.append({"params": head_params_no_wd, "lr": LR_HEAD, "weight_decay": 0.0, "name": "head_no_wd"})
    if backbone_params_wd:
        param_groups.append({"params": backbone_params_wd, "lr": LR_BACKBONE, "weight_decay": 1e-4, "name": "backbone_wd"})
    if backbone_params_no_wd:
        param_groups.append({"params": backbone_params_no_wd, "lr": LR_BACKBONE, "weight_decay": 0.0, "name": "backbone_no_wd"})

    if len(param_groups) == 0:
        print("Warning: no params found for optimizer; falling back to all params.")
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model_obj.parameters()), lr=LR_HEAD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    optimizer_local = torch.optim.SGD(param_groups, momentum=MOMENTUM)
    return optimizer_local


def set_backbone_requires_grad(model_obj, flag: bool):
    if not hasattr(model_obj, "dpt_backbone"):
        print("Warning: model has no dpt_backbone attribute")
        return
    for name, p in model_obj.dpt_backbone.named_parameters():
        p.requires_grad = bool(flag)


def train():
    
    dataset = FoodSeg103Dataset(
        root="data/FoodSeg103",
        transform=image_transform,
        target_transform=mask_transform,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)

    category_names = dataset.labels
    print("categories:", category_names[:10], " ... total:", len(category_names))
    num_classes = len(category_names)
    print("num_classes:", num_classes, "dataset size:", len(dataset))

    
    model = LSegModel(model_name="ViT-B/32").to(DEVICE).float()

    # Precompute text embeddings once (training prompts)
    with torch.no_grad():
        text_features = build_prompt_features(model, category_names, PROMPT_TEMPLATES)  # [N, C] on DEVICE
    print("text_features shape:", text_features.shape)

    # Freeze backbone initially for stability
    set_backbone_requires_grad(model, False)
    print("Backbone initially frozen.")

    
    criterion = SegmentationLoss(ignore_index=255)

    
    optimizer = build_optimizer(model)
    steps_per_epoch = max(1, len(dataloader))
    total_iters = NUM_EPOCHS * steps_per_epoch
    scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=0.9)

    best_miou = 0.0
    global_iter = 0
    start_epoch = 0

    # Warmup state for backbone after unfreeze
    backbone_warmup_state = {"active": False, "iters_done": 0, "warmup_iters": 0}

    # optionally resume
    if RESUME_FROM is not None and os.path.exists(RESUME_FROM):
        print(f"Resuming from checkpoint: {RESUME_FROM}")
        start_epoch, best_miou = load_checkpoint(RESUME_FROM, model, optimizer=optimizer, scheduler=scheduler, scaler=None, device=DEVICE, strict_model=False)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # set train/eval mode for visual depending on freeze status
        if epoch < BACKBONE_UNFREEZE_EPOCH:
            model.dpt_backbone.visual.eval()
        else:
            model.dpt_backbone.visual.train()

        model.train()
        epoch_loss = 0.0

        # Unfreeze backbone at specified epoch (and rebuild optimizer with small backbone LR)
        if epoch == BACKBONE_UNFREEZE_EPOCH:
            print(f"==> Unfreezing backbone at epoch {epoch}")
            set_backbone_requires_grad(model, True)  # full unfreeze per plan
            # rebuild optimizer so backbone params get LR_BACKBONE
            optimizer = build_optimizer(model)
            # scheduler must be re-created to match new optimizer param groups
            scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=0.9)

            # prepare backbone warmup (linear from 0.01 * LR_BACKBONE -> LR_BACKBONE)
            warmup_epochs = 3
            warmup_iters = min(1000, warmup_epochs * steps_per_epoch)
            backbone_warmup_state["active"] = True
            backbone_warmup_state["iters_done"] = 0
            backbone_warmup_state["warmup_iters"] = warmup_iters
            # initialize backbone lr to 1% of LR_BACKBONE
            for pg in optimizer.param_groups:
                if pg.get("name", "").startswith("backbone"):
                    pg["lr"] = LR_BACKBONE * 0.01
            print(f"Backbone warmup: {warmup_iters} iters; starting backbone lr = {LR_BACKBONE * 0.01}")

        total_inter = torch.zeros(num_classes).to(device)
        total_union = torch.zeros(num_classes).to(device)

        for batch_idx, (images, masks) in enumerate(dataloader):
            model.train()
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            # (optional) if warmup active, compute current backbone lr scale before step
            if backbone_warmup_state["active"]:
                it_done = backbone_warmup_state["iters_done"]
                w = backbone_warmup_state["warmup_iters"]
                # linear scale from 0.01 -> 1.0
                scale = 0.01 + 0.99 * (min(it_done, w) / max(1, w))
                for pg in optimizer.param_groups:
                    if pg.get("name", "").startswith("backbone"):
                        pg["lr"] = LR_BACKBONE * scale

            optimizer.zero_grad()
            outputs = model(images, text_features=text_features)  # [B, N, H, W]
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            # step warmup counter (after step)
            if backbone_warmup_state["active"]:
                backbone_warmup_state["iters_done"] += 1
                if backbone_warmup_state["iters_done"] >= backbone_warmup_state["warmup_iters"]:
                    # warmup finished -> set backbone param groups lr to LR_BACKBONE exactly
                    for pg in optimizer.param_groups:
                        if pg.get("name", "").startswith("backbone"):
                            pg["lr"] = LR_BACKBONE
                    backbone_warmup_state["active"] = False
                    print("Backbone warmup finished; backbone lr set to", LR_BACKBONE)

            # scheduler per-iteration
            scheduler.step()
            global_iter += 1
            epoch_loss += loss.item()

            # compute preds for mIoU
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]
            inter, union = intersection_and_union(preds, masks, num_classes)
            total_inter += inter
            total_union += union

            if (batch_idx) % 25 == 0:
                unique_classes = torch.unique(preds)
                iou = total_inter / (total_union + 1e-10)
                miou = iou.mean().item()
                try:
                    lr_now = [pg["lr"] for pg in optimizer.param_groups]
                except Exception:
                    lr_now = scheduler.get_last_lr()
                print(f"[Epoch {epoch} Batch {batch_idx}] Loss: {loss.item():.4f} miou: {miou:.4f} | lr: {lr_now} | unique_classes: {len(unique_classes)}")
            if (batch_idx +1) %150==0:
                evaluate_zero_shot(model)


        # compute epoch mIoU
        avg_loss = epoch_loss / float(steps_per_epoch)
        iou = total_inter / (total_union + 1e-10)
        miou = iou.mean().item()

        # print per-epoch stats
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | mIoU (train): {miou:.4f}")

        # save checkpoint if improved or periodically
        if (epoch + 1) % SAVE_EVERY == 2 or miou > best_miou:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/lseg_epoch{epoch+1}.pt"
            save_checkpoint(save_path, model, optimizer=optimizer, scheduler=scheduler, epoch=epoch+1, miou=miou, scaler=None, keep_last_k=None)
            print(f"Saved: {save_path}")
            if miou > best_miou:
                best_miou = miou

    print("Training finished. Best mIoU:", best_miou)

if __name__ == "__main__":
    train()
