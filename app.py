# --------------------------------------------------------------
#  app.py  –  Streamlit Image Annotation Tool (Pascal VOC / YOLO only)
# --------------------------------------------------------------

# ------------------------------------------------------------------
#  Compatibility shim – restores the removed `image_to_url` helper
# ------------------------------------------------------------------
import streamlit as st
import base64
from io import BytesIO
import importlib
import sys
from typing import Any

def _pil_to_data_url(pil_img, fmt: str = "PNG") -> str:
    """Encode a Pillow image as a base‑64 data‑URL (exactly what the old helper returned)."""
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def _install_image_to_url(target: Any) -> None:
    """
    Attach a tiny ``image_to_url`` function to *target* (module, package, or object).
    The function accepts either a Pillow image or a ready‑made data‑URL string.
    """
    if hasattr(target, "image_to_url"):
        # Already present – nothing to do.
        return

    def _image_to_url(img, *_, **__):
        # The component may already have handed us a data‑URL string.
        if isinstance(img, str):
            return img
        # Otherwise we assume a Pillow Image.
        return _pil_to_data_url(img)

    setattr(target, "image_to_url", _image_to_url)   # type: ignore


# ------------------------------------------------------------------
#  1️⃣  Patch the public `streamlit` package (the object the component imports as ``import streamlit as st``)
# ------------------------------------------------------------------
_install_image_to_url(st)

# ------------------------------------------------------------------
#  2️⃣  Patch the historic private module (Streamlit ≤ 1.29) if it still exists.
# ------------------------------------------------------------------
try:
    _elements_mod = importlib.import_module("streamlit.elements.image")
    _install_image_to_url(_elements_mod)
except Exception:   # pragma: no cover
    pass

# ------------------------------------------------------------------
#  3️⃣  Patch the newer internal location (Streamlit 1.30+ moved many internals under ``streamlit.runtime``).
# ------------------------------------------------------------------
try:
    _runtime_mod = importlib.import_module("streamlit.runtime")
    _install_image_to_url(_runtime_mod)
except Exception:   # pragma: no cover
    pass

# ------------------------------------------------------------------
#  4️⃣  (Optional) silence IDE warnings about a non‑existent ``streamlit.image`` module.
# ------------------------------------------------------------------
_fake_mod_name = "streamlit.image"
if _fake_mod_name not in sys.modules:
    fake_mod = type(sys)('streamlit.image')
    _install_image_to_url(fake_mod)
    sys.modules[_fake_mod_name] = fake_mod

# ------------------------------------------------------------------
#  Import the drawable‑canvas component (after the shim!)
# ------------------------------------------------------------------
from streamlit_drawable_canvas import st_canvas

# ------------------------------------------------------------------
#  Standard imports
# ------------------------------------------------------------------
import json
import os
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from PIL import Image

# ------------------------------------------------------------------
#  Helper – Pillow → data‑URL (kept for external use)
# ------------------------------------------------------------------
def pil_to_data_url(pil_img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a Pillow image as a base‑64 data‑URL."""
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


# ------------------------------------------------------------------
#  Misc helpers (size, export formats, zip creation…)
# ------------------------------------------------------------------
def get_image_size(image_path: str) -> tuple[int, int]:
    """Return (width, height) of an image file."""
    with Image.open(image_path) as img:
        return img.width, img.height


def create_pascal_voc_xml(image_name, width, height, boxes) -> str:
    """Return a Pascal VOC XML string for a single image."""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = image_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for box in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "object"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = int(box["x"])
        ymin = int(box["y"])
        xmax = int(box["x"] + box["width"])
        ymax = int(box["y"] + box["height"])
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
    return ET.tostring(annotation, encoding="unicode")


def create_yolo_txt(width, height, boxes) -> str:
    """Return a YOLO‑txt string for a single image."""
    lines = []
    for box in boxes:
        cx = (box["x"] + box["width"] / 2) / width
        cy = (box["y"] + box["height"] / 2) / height
        w = box["width"] / width
        h = box["height"] / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def zip_bytes(file_dict: Dict[str, bytes]) -> BytesIO:
    """Create an in‑memory ZIP from a dict {filename: bytes}."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in file_dict.items():
            zf.writestr(fname, fbytes)
    buffer.seek(0)
    return buffer


# ------------------------------------------------------------------
#  Helper – scale boxes back to the *original* image size
# ------------------------------------------------------------------
def _scale_boxes_to_original(img_name: str, boxes: List[dict]) -> List[dict]:
    """Rescale canvas boxes to the size of the original file on disk."""
    orig_path = os.path.join(st.session_state.temp_dir, img_name)
    orig_w, orig_h = get_image_size(orig_path)

    canvas_w, canvas_h = st.session_state.pil_images[img_name].size

    fx = orig_w / canvas_w
    fy = orig_h / canvas_h

    return [
        {
            "x": box["x"] * fx,
            "y": box["y"] * fy,
            "width": box["width"] * fx,
            "height": box["height"] * fy,
        }
        for box in boxes
    ]


# ------------------------------------------------------------------
#  Session‑state defaults (persist across reruns)
# ------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "initialized": False,
        "temp_dir": None,
        "image_files": [],   # full paths
        "image_names": [],   # just the file names
        "pil_images": {},    # name → (maybe resized) Pillow image
        "current_idx": 0,
        "annotations": {},   # name → list[boxes] (raw for export)
        "canvas_json": {},   # name → raw canvas JSON (for preload)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ------------------------------------------------------------------
#  Sidebar – ZIP upload & Reset
# ------------------------------------------------------------------
st.sidebar.title("🔧 Controls")

uploaded_zip = st.sidebar.file_uploader(
    "Upload a ZIP containing PNG/JPG/JPEG images",
    type="zip",
    help="The zip may contain sub‑folders; only image files are extracted.",
)

# ----- unzip & index images -----------------------------------------
if uploaded_zip is not None and not st.session_state.initialized:
    tmp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = tmp_dir

    with zipfile.ZipFile(uploaded_zip) as zf:
        zf.extractall(tmp_dir)

    img_paths = []
    for root, _dirs, files in os.walk(tmp_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_paths.append(os.path.join(root, f))
    img_paths.sort()

    st.session_state.image_files = img_paths
    st.session_state.image_names = [os.path.basename(p) for p in img_paths]

    # Load each image once – down‑scale only if it is wider than 1024 px
    MAX_W = 700
    for path, name in zip(img_paths, st.session_state.image_names):
        pil = Image.open(path).convert("RGB")
        if pil.width > MAX_W:
            ratio = MAX_W / pil.width
            pil = pil.resize(
                (MAX_W, int(pil.height * ratio)), Image.Resampling.LANCZOS
            )
        st.session_state.pil_images[name] = pil

    st.session_state.annotations = {n: [] for n in st.session_state.image_names}
    st.session_state.canvas_json = {}
    st.session_state.current_idx = 0
    st.session_state.initialized = True
    st.success(f"✅ Loaded {len(img_paths)} image(s)")

# ----- Reset button -------------------------------------------------
if st.sidebar.button("🔄 Reset Application", type="primary"):
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_session_state()
    st.rerun()   # new API (replaces experimental_rerun)


# ------------------------------------------------------------------
#  Main UI – only after a zip has been loaded
# ------------------------------------------------------------------
st.title("🖼️ Image Annotation Tool")

if not st.session_state.initialized:
    st.info("👈 Upload a ZIP of images in the sidebar to begin.")
    st.stop()

# ------------------------------------------------------------------
# Navigation buttons (← Previous / Next →)
# ------------------------------------------------------------------
nav_left, nav_mid, nav_right = st.columns([1, 2, 1])

with nav_left:
    if st.button("← Previous"):
        if st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1

with nav_mid:
    st.markdown(
        f"**Image {st.session_state.current_idx + 1} / {len(st.session_state.image_names)}**"
    )

with nav_right:
    if st.button("Next →"):
        if st.session_state.current_idx < len(st.session_state.image_names) - 1:
            st.session_state.current_idx += 1
# ------------------------------------------------------------------
#  Load the current image (Pillow) + optional preload JSON
# ------------------------------------------------------------------
cur_name = st.session_state.image_names[st.session_state.current_idx]
pil_image = st.session_state.pil_images[cur_name]

img_w, img_h = pil_image.width, pil_image.height

st.subheader(f"📷 {cur_name}")

preload_json = st.session_state.canvas_json.get(cur_name, {})

# --------------------------------------------------------------
#  Canvas – unique key per navigation so `initial_drawing` is honoured
# --------------------------------------------------------------
canvas_key = f"canvas_{cur_name}_{st.session_state.current_idx}"

canvas_result = st_canvas(
    fill_color="transparent",
    stroke_width=2,
    stroke_color="#ff0000",
    background_image=pil_image,
    height=img_h,
    width=img_w,
    drawing_mode="rect",
    key=canvas_key,
    initial_drawing=preload_json,
    update_streamlit=False,
)

# ------------------------------------------------------------------
# Store / update bounding boxes only when the user changes something
# ------------------------------------------------------------------
if canvas_result.json_data is not None:
    new_boxes = [
        {
            "x": obj["left"],
            "y": obj["top"],
            "width": obj["width"],
            "height": obj["height"],
        }
        for obj in canvas_result.json_data.get("objects", [])
        if obj.get("type") == "rect"
    ]

    st.session_state.annotations[cur_name] = new_boxes
    st.session_state.canvas_json[cur_name] = canvas_result.json_data


# ------------------------------------------------------------------
# Export block – annotations only OR everything together
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("📤 Export")

export_fmt = st.selectbox(
    "Choose annotation format",
    ["Pascal VOC (XML)", "YOLO (TXT)"],
    index=0,
)

# ------------------- 1️⃣ Export only the annotation files -------------------
if st.button("Export Annotations (no images)"):
    export_files: Dict[str, bytes] = {}

    if export_fmt == "Pascal VOC (XML)":
        for img_name, boxes in st.session_state.annotations.items():
            boxes = _scale_boxes_to_original(img_name, boxes)
            w, h = get_image_size(os.path.join(st.session_state.temp_dir, img_name))
            xml = create_pascal_voc_xml(img_name, w, h, boxes)
            export_files[f"{Path(img_name).stem}.xml"] = xml.encode("utf-8")
    else:   # YOLO (TXT)
        for img_name, boxes in st.session_state.annotations.items():
            boxes = _scale_boxes_to_original(img_name, boxes)
            w, h = get_image_size(os.path.join(st.session_state.temp_dir, img_name))
            txt = create_yolo_txt(w, h, boxes)
            export_files[f"{Path(img_name).stem}.txt"] = txt.encode("utf-8")

    zip_buf = zip_bytes(export_files)
    st.download_button(
        label="💾 Download annotations ZIP",
        data=zip_buf,
        file_name="annotations.zip",
        mime="application/zip",
    )

# ------------------- 2️⃣ Export everything (images + annotations) ----------
if st.button("Export All (images + annotations)"):
    export_files: Dict[str, bytes] = {}

    # Helper that adds the annotation file for a *single* image
    def _add_annotation(img_name: str, boxes: List[dict]) -> None:
        img_path = os.path.join(st.session_state.temp_dir, img_name)
        w, h = get_image_size(img_path)

        if export_fmt == "Pascal VOC (XML)":
            xml = create_pascal_voc_xml(img_name, w, h, boxes)
            export_files[f"{Path(img_name).stem}.xml"] = xml.encode("utf-8")
        else:   # YOLO (TXT)
            txt = create_yolo_txt(w, h, boxes)
            export_files[f"{Path(img_name).stem}.txt"] = txt.encode("utf-8")

    # Add every original image (as‑is) + its annotation
    for img_path in st.session_state.image_files:
        img_name = os.path.basename(img_path)

        # copy the raw image file into the ZIP
        with open(img_path, "rb") as f:
            export_files[img_name] = f.read()

        boxes = _scale_boxes_to_original(img_name, st.session_state.annotations[img_name])
        _add_annotation(img_name, boxes)

    zip_buf = zip_bytes(export_files)
    st.download_button(
        label="💾 Download all files ZIP",
        data=zip_buf,
        file_name="annotations_and_images.zip",
        mime="application/zip",
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption(
    """
    **Note:** All data lives only in the current browser session.
    Close the tab or press “Reset” to discard your work.
    """
)