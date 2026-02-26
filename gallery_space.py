import gradio as gr
from pathlib import Path
import os
import shutil
from huggingface_hub import HfApi, snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")

api = HfApi()
username = api.whoami(token=HF_TOKEN)["name"]
repo = "sd_out"
user_repo = f"{username}/{repo}"


def refresh_images():
    try:
        shutil.rmtree(os.environ.get("IMAGE_DIR", ""))
    except:
        pass

    image_dir = Path(
        snapshot_download(repo_id=user_repo, repo_type="dataset", token=HF_TOKEN)
    )
    os.environ["IMAGE_DIR"] = str(image_dir)

    image_files = list(Path(image_dir).rglob("*.[pjw]*[npjg]*[ge]*"))

    local_dir = Path("images")
    local_dir.mkdir(exist_ok=True)

    copied_files = []
    for img in image_files:
        dst = local_dir / img.name
        shutil.copy(img, dst)
        copied_files.append(str(dst))

    return copied_files


# ✅ CSS scroll fix
custom_css = """
#scroll-gallery {
    overflow-y: auto !important;
    overflow-x: auto !important;
}
#scroll-gallery img {
    max-height: none !important;
}
"""

with gr.Blocks(
    analytics_enabled=False,
    title="Image Gallery",
    theme="NoCrypt/miku",
    css=custom_css,
) as demo:

    gr.HTML("<center><h1>Image Gallery</h1></center>")

    submit = gr.Button("Refresh", variant="primary")

    gallery = gr.Gallery(
        value=[],
        columns=4,
        show_label=False,
        height=800,
        object_fit="contain",
        elem_id="scroll-gallery",  # ✅ penting
    )

    submit.click(refresh_images, outputs=[gallery])

demo.launch(debug=True)
