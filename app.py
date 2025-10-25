import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, os, gradio as gr
from torchvision import transforms
from src.models.lseg_model import LSegModel  # import your model

# ---------------------------
# Settings
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoints/Lseg.pt"  # change if needed
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a picture of {}",
    "a close-up of {}",
    "a plate of {}",
    "a dish containing {}",
]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ---------------------------
# Utils
# ---------------------------

def build_prompt_features(model, categories, templates):
    """Encode text prompts and average across templates."""
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for cat in categories:
            prompts = [t.format(cat) for t in templates]
            feats = model.text_encoder(prompts).to(DEVICE).float()
            mean_feat = feats.mean(dim=0, keepdim=True)
            all_embeddings.append(mean_feat)
    text_features = torch.cat(all_embeddings, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha=0.6):
    """Overlay mask on image with clear color separation."""
    num_classes = int(mask.max()) + 1
    colors = np.array([
        plt.cm.get_cmap("tab20")(i % 20)[:3] for i in range(num_classes)
    ]) * 255
    colors = colors.astype(np.uint8)
    mask_rgb = colors[mask]
    overlay = (image * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return overlay, colors

def create_legend(colors, class_names):
    """Return a small legend image."""
    fig, ax = plt.subplots(figsize=(4, 0.3 * len(class_names)))
    patches = [mpatches.Patch(color=colors[i]/255, label=class_names[i])
               for i in range(len(class_names))]
    ax.legend(handles=patches, loc='center left', frameon=False)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# ---------------------------
# Load Model (once)
# ---------------------------

print("Loading model...")
model = LSegModel(model_name="ViT-B/32").to(DEVICE).float()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")

# ---------------------------
# Inference Function
# ---------------------------

def infer(image, class_text):
    if image is None or not class_text.strip():
        raise gr.Error("Please upload an image and enter class names.")

    class_names = [c.strip() for c in class_text.split(',') if c.strip()]
    img_pil = image.convert("RGB")
    img_tensor = image_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        text_features = build_prompt_features(model, class_names, PROMPT_TEMPLATES).to(DEVICE)
        outputs = model(img_tensor, text_features=text_features)
        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    overlay, colors = overlay_mask(img_np, preds)

    # Create visual legend
    legend_img = create_legend(colors, class_names)

    # Save files
    os.makedirs("outputs", exist_ok=True)
    mask_path = "outputs/mask.png"
    overlay_path = "outputs/overlay.png"
    Image.fromarray(colors[preds]).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)

    return colors[preds], overlay, legend_img, mask_path, overlay_path

# ---------------------------
# Gradio UI
# ---------------------------

title = "🍱 LSeg Segmentation Demo"
description = """
Upload a food image and list your classes (comma separated).  
The app will produce:
- A **colored segmentation mask**
- An **overlayed image**
- A **legend** showing class-color mapping
"""

iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Class Names (comma-separated)", placeholder="e.g. lemon, tomato, white rice, lettuce")
    ],
    outputs=[
        gr.Image(label="Predicted Mask"),
        gr.Image(label="Overlay"),
        gr.Image(label="Legend"),
        gr.File(label="Download Mask"),
        gr.File(label="Download Overlay")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

iface.launch(server_name="127.0.0.1", server_port=7860, debug=True)
