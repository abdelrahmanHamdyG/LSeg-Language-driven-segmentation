import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import gradio as gr
import os

# Try to import your model, with fallback for demo purposes
try:
    from src.models.lseg_model import LSegModel
    MODEL_AVAILABLE = True
except ImportError:
    print("⚠️  LSegModel not found, using dummy model for demo")
    MODEL_AVAILABLE = False
    # Create a dummy model for testing
    class LSegModel(torch.nn.Module):
        def __init__(self, model_name="ViT-B/32"):
            super().__init__()
            self.model_name = model_name
        def forward(self, x, text_features=None):
            # Return dummy segmentation map
            batch_size, _, h, w = x.shape
            return torch.randn(batch_size, 5, h, w)
        def text_encoder(self, prompts):
            # Return dummy text features
            return torch.randn(len(prompts), 512)

# ---------------------------
# Settings
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoints/lseg_epoch36.pt"

PROMPT_TEMPLATES = [
    "a photo of {}",
    "a picture of {}",
    "a close-up of {}",
    "a plate of {}",
    "a dish containing {}",
]

DEFAULT_CLASS_NAMES = "others, peas, tomato, white rice, chicken"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ---------------------------
# Image transform
# ---------------------------
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# ---------------------------
# Model loading
# ---------------------------
def load_model():
    """Load the model with error handling"""
    if not MODEL_AVAILABLE:
        print("⚠️  Using dummy model - LSegModel not available")
        return LSegModel()
    
    try:
        model = LSegModel(model_name="ViT-B/32").to(DEVICE).float()
        
        # Check if checkpoint exists
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {CHECKPOINT_PATH}")
        else:
            print(f"⚠️  Checkpoint not found at {CHECKPOINT_PATH}, using untrained model")
            
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  Falling back to dummy model")
        return LSegModel()

# Load model globally
model = load_model()

# ---------------------------
# Build prompt features
# ---------------------------
def build_prompt_features(categories, templates):
    """Encode multiple text templates per class."""
    all_embeddings = []
    with torch.no_grad():
        for cat in categories:
            prompts = [t.format(cat) for t in templates]
            try:
                feats = model.text_encoder(prompts).to(DEVICE).float()
                mean_feat = feats.mean(dim=0, keepdim=True)
                all_embeddings.append(mean_feat)
            except Exception as e:
                print(f"❌ Error encoding prompt for '{cat}': {e}")
                # Create dummy features
                dummy_feat = torch.randn(1, 512).to(DEVICE).float()
                all_embeddings.append(dummy_feat)
    
    if not all_embeddings:
        raise ValueError("No embeddings were created")
        
    text_features = torch.cat(all_embeddings, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# ---------------------------
# Generate consistent, distinct colors
# ---------------------------
def generate_distinct_colors(num_classes):
    """Generate visually distinct colors for each class."""
    cmap = plt.get_cmap("tab20", num_classes)
    colors = (np.array([cmap(i)[:3] for i in range(num_classes)]) * 255).astype(np.uint8)
    return colors

# ---------------------------
# Overlay mask on image
# ---------------------------
def overlay_mask(image: np.ndarray, mask: np.ndarray, colors: np.ndarray, alpha=0.6):
    """Overlay the segmentation mask on top of the image."""
    mask_rgb = colors[mask]
    overlay = (image * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return overlay

# ---------------------------
# Process class names input
# ---------------------------
def parse_class_names(class_names_str):
    """Parse comma-separated class names into a list."""
    classes = [name.strip() for name in class_names_str.split(",") if name.strip()]
    return classes if classes else ["background"]

# ---------------------------
# Main inference function for Gradio
# ---------------------------
def segment_image(input_image, class_names_str):
    """
    Main segmentation function for Gradio interface.
    """
    try:
        # Validate inputs
        if input_image is None:
            return None, None, None, "❌ Please upload an image"
        
        if not class_names_str or not class_names_str.strip():
            return None, None, None, "❌ Please provide class names"
        
        # Parse class names
        class_names = parse_class_names(class_names_str)
        if len(class_names) == 0:
            return None, None, None, "❌ Please provide at least one class name"
        
        print(f"🔍 Processing image with {len(class_names)} classes: {class_names}")
        
        # Convert Gradio image to PIL
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image.astype('uint8'))
        else:
            pil_image = input_image
        
        # Ensure RGB
        pil_image = pil_image.convert("RGB")
        original_size = pil_image.size
        
        # Transform image
        img_tensor = image_transform(pil_image).unsqueeze(0).to(DEVICE)
        
        # Build text features
        with torch.no_grad():
            text_features = build_prompt_features(class_names, PROMPT_TEMPLATES)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor, text_features=text_features)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        # Resize original image to match prediction size for display
        display_size = (IMG_SIZE, IMG_SIZE)
        img_np = np.array(pil_image.resize(display_size))
        
        # Generate colors
        colors = generate_distinct_colors(len(class_names))
        
        # Create mask and overlay
        mask_image = colors[preds]
        overlay_image = overlay_mask(img_np, preds, colors)
        
        # Convert to PIL for Gradio
        original_pil = Image.fromarray(img_np)
        mask_pil = Image.fromarray(mask_image)
        overlay_pil = Image.fromarray(overlay_image)
        
        # Create legend image
        legend_pil = create_legend_image(class_names, colors)
        
        success_msg = f"✅ Segmentation complete! Processed {len(class_names)} classes"
        return original_pil, mask_pil, overlay_pil, legend_pil, success_msg
        
    except Exception as e:
        error_msg = f"❌ Error during segmentation: {str(e)}"
        print(error_msg)
        return None, None, None, None, error_msg

def create_legend_image(class_names, colors):
    """Create a legend image showing class colors and names."""
    fig, ax = plt.subplots(figsize=(6, len(class_names) * 0.5))
    ax.axis('off')
    
    # Create legend patches
    for i, (name, color) in enumerate(zip(class_names, colors/255.0)):
        ax.add_patch(plt.Rectangle((0.1, i*0.8), 0.2, 0.6, color=color))
        ax.text(0.4, i*0.8 + 0.3, name, fontsize=12, va='center')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(class_names) * 0.8)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    legend_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    legend_array = legend_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return Image.fromarray(legend_array)

# ---------------------------
# Create Gradio interface
# ---------------------------
def create_interface():
    with gr.Blocks(
        title="LSeg Image Segmentation",
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-image {
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🖼️ LSeg Image Segmentation
            Upload an image and specify comma-separated class names to perform zero-shot segmentation.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    image_input = gr.Image(
                        label="📷 Upload Image", 
                        type="numpy",
                        height=300,
                        sources=["upload", "clipboard"]
                    )
                    class_names_input = gr.Textbox(
                        label="🏷️ Class Names (comma-separated)",
                        value=DEFAULT_CLASS_NAMES,
                        placeholder="e.g., sky, tree, building, road, person",
                        lines=2
                    )
                    process_btn = gr.Button(
                        "🚀 Segment Image", 
                        variant="primary",
                        size="lg"
                    )
                
                # Status message
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📸 Original"):
                        original_output = gr.Image(
                            label="Original Image", 
                            height=300,
                            interactive=False,
                            elem_classes="output-image"
                        )
                    
                    with gr.TabItem("🎭 Segmentation Mask"):
                        mask_output = gr.Image(
                            label="Segmentation Mask", 
                            height=300,
                            interactive=False,
                            elem_classes="output-image"
                        )
                    
                    with gr.TabItem("🔍 Overlay Result"):
                        overlay_output = gr.Image(
                            label="Overlay Result", 
                            height=300,
                            interactive=False,
                            elem_classes="output-image"
                        )
                
                with gr.Row():
                    legend_output = gr.Image(
                        label="Legend",
                        height=200,
                        interactive=False,
                        show_label=True
                    )
        
        # Examples
        gr.Markdown("### 📚 Examples")
        gr.Examples(
            examples=[
                ["examples/food.jpg", "others, peas, tomato, white rice, chicken"],
                ["examples/street.jpg", "sky, building, road, tree, car, person"],
            ],
            inputs=[image_input, class_names_input],
            outputs=[original_output, mask_output, overlay_output, legend_output, status_output],
            fn=segment_image,
            cache_examples=False,
            label="Click an example below to load it:"
        )
        
        # Event handling
        process_btn.click(
            fn=segment_image,
            inputs=[image_input, class_names_input],
            outputs=[original_output, mask_output, overlay_output, legend_output, status_output]
        )
        
        gr.Markdown(
            """
            ### 💡 Usage Tips:
            - Upload any image (JPG, PNG, etc.)
            - Enter class names as comma-separated values
            - The model will segment the image based on your specified classes
            - Results show in tabs: original image, segmentation mask, and overlay
            - Legend shows the color mapping for each class
            
            ### 🛠️ Technical Details:
            - Model: LSeg with CLIP ViT-B/32 backbone
            - Image size: 224×224 pixels
            - Device: {} 
            """.format(DEVICE)
        )
    
    return demo

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting LSeg Gradio Interface...")
    print(f"📱 Device: {DEVICE}")
    print(f"🖼️  Image size: {IMG_SIZE}×{IMG_SIZE}")
    print("🌐 Starting server...")
    
    demo = create_interface()
    
    demo.launch(
        server_name="127.0.0.1",  # Local only
        server_port=7860,
        share=False,
        show_error=True,
        debug=False,
        inbrowser=True  # Automatically open browser
    )