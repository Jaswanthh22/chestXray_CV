"""Simple desktop interface for chest X-ray predictions and Grad-CAM."""

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.keras"
IMAGE_SIZE = 224
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
]


class ChestXrayApp:
    def __init__(self, root):
        # Keep references to the window, model, and displayed images for later callbacks.
        self.root = root
        self.root.title("Chest X-ray Classifier")
        self.root.geometry("1040x760")
        self.root.minsize(850, 650)
        self.model = None
        self.original_photo = None
        self.cam_photo = None

        # Define reusable text styles and build the main window layout.
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))

        container = ttk.Frame(root, padding=22)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="Chest X-ray Classifier", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text="Choose an image to see model scores and a Grad-CAM overlay.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 14))
        self.top_result = tk.StringVar(value="Top finding: choose an image")
        ttk.Label(
            container,
            textvariable=self.top_result,
            font=("Segoe UI", 16, "bold"),
            foreground="#174ea6",
        ).pack(anchor="w", pady=(0, 14))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 14))
        self.choose_button = ttk.Button(controls, text="Choose X-ray", command=self.choose_image)
        self.choose_button.pack(side="left")
        self.status = tk.StringVar(value=f"Model: {MODEL_PATH.name}")
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=12)

        content = ttk.Frame(container)
        content.pack(fill="both", expand=True)
        image_panel = ttk.Frame(content)
        image_panel.pack(side="left", fill="both", expand=True)
        score_panel = ttk.Frame(content, padding=(22, 0, 0, 0), width=260)
        score_panel.pack(side="right", fill="y")
        score_panel.pack_propagate(False)

        images = ttk.Frame(image_panel)
        images.pack(fill="both", expand=True)
        for column in range(2):
            images.columnconfigure(column, weight=1, uniform="images")
        images.rowconfigure(1, weight=1)
        ttk.Label(images, text="Selected image", style="Section.TLabel").grid(row=0, column=0, pady=(0, 8))
        ttk.Label(images, text="Grad-CAM overlay", style="Section.TLabel").grid(row=0, column=1, pady=(0, 8))
        self.original_view = ttk.Label(images, text="Choose an image to begin", anchor="center")
        self.original_view.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.cam_view = ttk.Label(images, text="Grad-CAM will appear here", anchor="center")
        self.cam_view.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        ttk.Label(score_panel, text="Finding scores", style="Section.TLabel").pack(anchor="w", pady=(0, 12))
        self.score_labels = {}
        for name in CLASSES:
            row = ttk.Frame(score_panel)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=name).pack(side="left")
            score = ttk.Label(row, text="?", width=7, anchor="e")
            score.pack(side="right")
            self.score_labels[name] = score

        ttk.Label(
            container,
            text="Educational use only. Model scores are not calibrated diagnoses; Grad-CAM does not confirm correctness.",
            wraplength=960,
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(14, 0))

    def choose_image(self):
        """Let the user select an image, preview it, and start background analysis."""
        # Restrict the picker to common image formats, while still offering all files.
        filename = filedialog.askopenfilename(
            title="Choose a chest X-ray",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            # Convert to RGB for consistent display, then shrink only the preview.
            with Image.open(filename) as source:
                image = source.convert("RGB")
                display_image = image.copy()
                display_image.thumbnail((440, 500))
                self.original_photo = ImageTk.PhotoImage(display_image)
                self.original_view.configure(image=self.original_photo, text="")
        except Exception as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return

        # Disable repeated requests while inference runs on a worker thread.
        self.choose_button.configure(state="disabled")
        self.status.set(f"Analyzing {Path(filename).name}?")
        self.top_result.set("Top confidence: calculating?")
        for label in self.score_labels.values():
            label.configure(text="?")
        threading.Thread(target=self._analyze, args=(filename,), daemon=True).start()

    def _analyze(self, filename):
        """Load the model if needed, predict scores, and build a Grad-CAM overlay."""
        try:
            import cv2
            import tensorflow as tf

            # Load the saved Keras model lazily so startup remains fast.
            if self.model is None:
                if not MODEL_PATH.is_file():
                    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

            # Match the model's expected RGB, 224x224, [0, 1] input format.
            with Image.open(filename) as source:
                resized = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            batch = np.asarray(resized, dtype=np.float32)[None, ...] / 255.0
            # Predict all finding scores and explain the highest-scoring class.
            scores = self.model.predict(batch, verbose=0)[0]
            if len(scores) != len(CLASSES):
                raise ValueError(f"Expected {len(CLASSES)} outputs, got {len(scores)}")
            index = int(np.argmax(scores))
            # Resize and blend the class activation map with the input image.
            heatmap = self._gradcam(tf, batch, index)
            heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
            heatmap_color = (plt_jet(heatmap) * 255).astype(np.uint8)
            base = (batch[0] * 255).astype(np.uint8)
            overlay = np.clip(base * 0.6 + heatmap_color * 0.4, 0, 255).astype(np.uint8)
            self.root.after(0, self._show_results, scores, index, overlay)
        except Exception as exc:
            self.root.after(0, self._show_error, exc)

    def _gradcam(self, tf, batch, class_index):
        """Compute a Grad-CAM heatmap for one output class of the DenseNet model."""
        # Expose the final convolutional features and the backbone output together.
        backbone = self.model.get_layer("densenet121")
        conv_layer = backbone.get_layer("relu")
        grad_model = tf.keras.models.Model(backbone.input, [conv_layer.output, backbone.output])
        tensor = tf.convert_to_tensor(batch)
        # Replay the classifier head so gradients can be taken for the chosen score.
        with tf.GradientTape() as tape:
            conv_output, backbone_output = grad_model(tensor, training=False)
            output = backbone_output
            passed_backbone = False
            for layer in self.model.layers:
                if layer.name == backbone.name:
                    passed_backbone = True
                    continue
                if passed_backbone:
                    output = layer(output, training=False)
            score = output[:, class_index]
        grads = tape.gradient(score, conv_output)
        if grads is None:
            raise RuntimeError("Could not calculate Grad-CAM for this model.")
        # Average gradients per feature channel, then combine channels into a map.
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_output[0] * weights, axis=-1)
        cam = tf.maximum(cam, 0)
        return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()

    def _show_results(self, scores, index, overlay):
        """Update the Tkinter widgets with the overlay and per-class scores."""
        # Keep the PhotoImage objects on self so Tkinter does not garbage-collect them.
        cam_image = Image.fromarray(overlay)
        cam_image.thumbnail((440, 500))
        self.cam_photo = ImageTk.PhotoImage(cam_image)
        self.cam_view.configure(image=self.cam_photo, text="")
        for name, score in zip(CLASSES, scores):
            self.score_labels[name].configure(text=f"{score:.1%}")
        self.top_result.set(f"Top confidence: {CLASSES[index]} ? {scores[index]:.1%}")
        self.status.set(f"Grad-CAM shown for {CLASSES[index]} (highest score)")
        self.choose_button.configure(state="normal")

    def _show_error(self, error):
        """Report an analysis error and re-enable image selection."""
        self.status.set("Analysis failed")
        self.choose_button.configure(state="normal")
        messagebox.showerror("Prediction failed", str(error))


def plt_jet(values):
    """Return an RGB JET heatmap without requiring Matplotlib in the app."""
    import cv2
    # OpenCV produces BGR; convert to RGB and normalize channel values to [0, 1].
    return cv2.cvtColor(cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0


if __name__ == "__main__":
    # Create the desktop window and hand control to Tkinter's event loop.
    root = tk.Tk()
    app = ChestXrayApp(root)
    root.mainloop()
