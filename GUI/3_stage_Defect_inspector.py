
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tkinter import Tk, Label, Button, filedialog, Frame, Scale, HORIZONTAL, messagebox, Toplevel
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\Talifhani Nemaangani\Downloads\best (10).pt"
AE_PATH = r"C:\Users\Talifhani Nemaangani\Downloads\autoencoder_best_lat256_img512.pth"
CLS_PATH = r"C:\Users\Talifhani Nemaangani\Downloads\resnet_best_lat256_img512.pth"
RESULTS_DIR = r"C:\Users\Talifhani Nemaangani\Downloads\results"

ANALYTICS_PATH = os.path.join(RESULTS_DIR, "inspection_analytics.csv")
ANNOTATED_DIR = os.path.join(RESULTS_DIR, "annotated")
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
class DefectDetectionPipeline:
    def __init__(self, threshold_value=60):
        self.threshold_value = threshold_value

    def set_threshold(self, value):
        self.threshold_value = value

    def enhance_dark_spots_multiscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = gray.astype(np.float32)
        for s in [3, 7, 15, 31]:
            blur = cv2.GaussianBlur(gray, (s, s), 0).astype(np.float32)
            enhanced += (gray.astype(np.float32) - blur) * 0.5
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        enhanced = cv2.createCLAHE(4.0, (16, 16)).apply(enhanced)
        return enhanced

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        if contours:
            cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        object_only = cv2.bitwise_and(img, img, mask=mask)
        enhanced = self.enhance_dark_spots_multiscale(object_only)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256, img_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=256, num_classes=2, use_layernorm=False):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.use_ln = use_layernorm
        self.norm = nn.LayerNorm(512) if use_layernorm else nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        if not self.use_ln and x.dim() == 2 and x.size(0) == 1:
            pass  # skip BN for single-sample inference
        else:
            x = self.norm(x)
        x = self.relu(x)
        return self.fc2(x)

# ============================================================
# MAIN APPLICATION
# ============================================================
class InspectionApp:
    BOX_W, BOX_H = 360, 300
    THRESH_PRESETS = {"Side View": 60, "Top View": 90, "Bottom View": 120}

    def __init__(self, root):
        self.root = root
        self.root.title("Defect Inspector (3-Stage)")
        self.root.geometry("1250x740")
        self.root.config(bg="#1E1E1E")

        # === Load YOLO Detector ===
        self.model = YOLO(MODEL_PATH)

        # === Load Autoencoder ===
        ae_ckpt = torch.load(AE_PATH, map_location=DEVICE)
        self.autoencoder = Autoencoder(ae_ckpt["latent_dim"], ae_ckpt["img_size"]).to(DEVICE)
        state_dict = ae_ckpt["model_state_dict"]
        filtered_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}
        self.autoencoder.load_state_dict(filtered_dict, strict=False)
        self.autoencoder.eval()

        # === Load Classifier (auto-detect LN/BN) ===
        cls_ckpt = torch.load(CLS_PATH, map_location=DEVICE)
        state_keys = cls_ckpt["model_state_dict"].keys()
        use_ln = any("ln1" in k for k in state_keys)
        self.classifier = LatentClassifier(
            cls_ckpt.get("latent_dim", 256),
            cls_ckpt.get("num_classes", 2),
            use_layernorm=use_ln
        ).to(DEVICE)
        self.classifier.load_state_dict(cls_ckpt["model_state_dict"], strict=False)
        self.classifier.eval()
        print(f"[INFO] Loaded latent classifier using {'LayerNorm' if use_ln else 'BatchNorm'}")

        # === AE image transform ===
        self.ae_transform = transforms.Compose([
            transforms.Resize((ae_ckpt["img_size"], ae_ckpt["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.classes = ["defective", "non-defective"]

        self.pipe = DefectDetectionPipeline()
        self.selected_path = None
        self.proc_img = None

        # === Header ===
        Label(root, text="Defect Inspector", bg="#1E1E1E",
              fg="white", font=("Segoe UI", 18, "bold")).pack(pady=20)

        # === MAIN ROW ===
        main_row = Frame(root, bg="#1E1E1E")
        main_row.pack(pady=5)

        # LEFT COLUMN
        left_col = Frame(main_row, bg="#1E1E1E")
        left_col.grid(row=0, column=0, padx=40, sticky="n")
        self.frame_original = self._make_panel(left_col, "Original Image")
        Button(left_col, text="Select Image", width=14, command=self.select_image).pack(pady=20)

        # MIDDLE COLUMN
        mid_col = Frame(main_row, bg="#1E1E1E")
        mid_col.grid(row=0, column=1, padx=40, sticky="n")
        self.frame_preproc = self._make_panel(mid_col, "Preprocessed Image")

        views = Frame(mid_col, bg="#1E1E1E")
        views.pack(pady=6)
        for name in ["Side View", "Top View", "Bottom View"]:
            Button(views, text=name, width=12,
                   command=lambda n=name: self.set_predefined(n)).pack(padx=15, side="left", pady=20)

        slider_section = Frame(mid_col, bg="#1E1E1E")
        slider_section.pack(pady=10)
        Label(slider_section, text="Manual Threshold Adjustment:",
              fg="white", bg="#1E1E1E", font=("Segoe UI", 11, "bold")).pack()
        self.slider = Scale(slider_section, from_=0, to=255, orient=HORIZONTAL, length=220,
                            bg="#1E1E1E", fg="white", troughcolor="#0070CC",
                            highlightthickness=0, command=self.update_threshold)
        self.slider.set(60)
        self.slider.pack(pady=5)
        self.lbl_val = Label(slider_section, text="Current Threshold: 60 ",
                             bg="#1E1E1E", fg="white", font=("Segoe UI", 10))
        self.lbl_val.pack()

        # RIGHT COLUMN
        right_col = Frame(main_row, bg="#1E1E1E")
        right_col.grid(row=0, column=2, padx=40, sticky="n")
        self.frame_detect = self._make_panel(right_col, "Detected Output")
        Button(right_col, text="Inspect", width=14, command=self.run_detection).pack(pady=10)

        self.result_label = Label(right_col, text="Result: N/A", bg="#1E1E1E",
                                  fg="white", font=("Segoe UI", 12, "bold"))
        self.result_label.pack(pady=6)

        # LOWER BUTTONS
        bottom_row = Frame(root, bg="#1E1E1E")
        bottom_row.pack(side="bottom", pady=5)
        Button(bottom_row, text="View Analytics", width=16,
               command=self.open_analytics_window).pack(side="left", padx=12, pady=10)
        Button(bottom_row, text="Exit", width=14, command=root.quit).pack(side="left", padx=12, pady=10)

        if not os.path.exists(ANALYTICS_PATH):
            pd.DataFrame(columns=["timestamp", "filename", "defective"]).to_csv(ANALYTICS_PATH, index=False)

    # ============================================================
    # ANALYTICS WINDOW
    # ============================================================
    def open_analytics_window(self):
        if not os.path.exists(ANALYTICS_PATH):
            messagebox.showinfo("Info", "No analytics data yet.")
            return
        win = Toplevel(self.root)
        win.title("Inspection Analytics")
        win.geometry("600x270")
        win.config(bg="#1E1E1E")
        Label(win, text="Inspection Analytics", fg="#00BFFF", bg="#1E1E1E",
              font=("Segoe UI", 14, "bold")).pack(pady=10)
        frame = Frame(win, bg="#2D2D2D", highlightbackground="#555", highlightthickness=2)
        frame.pack(pady=8)
        df = pd.read_csv(ANALYTICS_PATH)
        total = len(df)
        defective = df["defective"].sum() if total > 0 else 0
        good = total - defective
        rate = (defective / total * 100) if total > 0 else 0
        Label(frame, text=f"Total Inspected: {total}", bg="#2D2D2D", fg="white").grid(row=0, column=0, padx=15, pady=6)
        Label(frame, text=f"Defective: {defective}", bg="#2D2D2D", fg="#FF6347").grid(row=0, column=1, padx=15, pady=6)
        Label(frame, text=f"Non-Defective: {good}", bg="#2D2D2D", fg="#32CD32").grid(row=0, column=2, padx=15, pady=6)
        Label(frame, text=f"Defect Rate: {rate:.2f}%", bg="#2D2D2D", fg="#00BFFF").grid(row=0, column=3, padx=15, pady=6)
        Button(win, text="Refresh", width=10, command=lambda: [win.destroy(), self.open_analytics_window()]).pack(pady=6)
        Button(win, text="Close", width=10, command=win.destroy).pack(pady=3)

    # ============================================================
    # CORE LOGIC
    # ============================================================
    def _make_panel(self, parent, title):
        frame = Frame(parent, width=self.BOX_W, height=self.BOX_H + 30,
                      bg="#2D2D2D", highlightbackground="#555", highlightthickness=2)
        frame.pack()
        frame.pack_propagate(False)
        Label(frame, text=title, bg="#2D2D2D", fg="white",
              font=("Segoe UI", 11, "bold")).pack(pady=3)
        lab = Label(frame, bg="#111")
        lab.pack(fill="both", expand=True, padx=4, pady=4)
        setattr(self, f"panel_{title.split()[0].lower()}", lab)
        return frame

    def set_predefined(self, view_name):
        val = self.THRESH_PRESETS.get(view_name)
        self.pipe.set_threshold(val)
        self.slider.set(val)
        self.lbl_val.config(text=f"Current Threshold: {val} (Preset: {view_name})")
        if self.selected_path:
            self.live_update_preprocessing()

    def update_threshold(self, val):
        val = int(val)
        self.pipe.set_threshold(val)
        self.lbl_val.config(text=f"Current Threshold: {val}")
        if self.selected_path:
            self.live_update_preprocessing()

    def select_image(self):
        p = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not p:
            return
        self.selected_path = p
        img = cv2.imread(p)
        self._show("original", img)
        self._show("preprocessed", np.zeros_like(img))
        self._show("detected", np.zeros_like(img))
        self.live_update_preprocessing()

    def live_update_preprocessing(self):
        img = cv2.imread(self.selected_path)
        out = self.pipe.preprocess_image(img)
        self.proc_img = out
        self._show("preprocessed", out)

    # ============================================================
    # UPDATED DETECTION METHOD (WITH DETECTOR OVERRIDE)
    # ============================================================
    def run_detection(self):
        if self.proc_img is None:
            messagebox.showinfo("Info", "Please select an image first.")
            return

        # === YOLO DETECTION ===
        results = self.model.predict(self.proc_img, verbose=False)
        annotated = results[0].plot()
        num_detections = len(results[0].boxes)

        # === CLASSIFICATION ===
        img_pil = Image.fromarray(cv2.cvtColor(self.proc_img, cv2.COLOR_BGR2RGB))
        tensor = self.ae_transform(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z = self.autoencoder(tensor)
            logits = self.classifier(z)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        classifier_label = self.classes[pred]
        final_text = ""
        color = "white"

        # === COMBINED LOGIC ===
        if num_detections > 0 and classifier_label == "non-defective":
            final_text = "DEFECT DETECTED (detector override)"
            color = "#FF6347"
            defective_found = 1
        elif classifier_label == "defective":
            final_text = f" DEFECTIVE ({conf:.2f})"
            color = "#FF6347"
            defective_found = 1
        else:
            final_text = f" NON-DEFECTIVE ({conf:.2f})"
            color = "#32CD32"
            defective_found = 0

        self.result_label.config(text=final_text, fg=color)

        # === SAVE ANNOTATED OUTPUT ===
        annotated_path = os.path.join(
            ANNOTATED_DIR, f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(annotated_path, annotated)
        self._show("detected", annotated)

        # === LOG TO ANALYTICS ===
        analytics = pd.read_csv(ANALYTICS_PATH)
        analytics.loc[len(analytics)] = [
            datetime.now(), os.path.basename(annotated_path), defective_found
        ]
        analytics.to_csv(ANALYTICS_PATH, index=False)

    def _show(self, which, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        tw, th = self.BOX_W, self.BOX_H - 20
        sc = min(tw / w, th / h)
        nw, nh = int(w * sc), int(h * sc)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        x, y = (tw - nw)//2, (th - nh)//2
        canvas[y:y+nh, x:x+nw] = resized
        imgtk = ImageTk.PhotoImage(Image.fromarray(canvas))
        getattr(self, f"panel_{which}").config(image=imgtk)
        getattr(self, f"panel_{which}").image = imgtk

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root = Tk()
    app = InspectionApp(root)
    root.mainloop()
