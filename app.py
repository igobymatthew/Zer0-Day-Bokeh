import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from dof_bokeh1 import BokehProcessor

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Zer0-Day Bokeh")
        self.geometry("1200x800")

        self.processor = None
        self.original_image = None
        self.processed_image = None
        self.focus_point = (0.5, 0.5)

        # Configure the main window to have a dark theme
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.configure(bg='#2E2E2E')
        self.style.configure('.', background='#2E2E2E', foreground='white')
        self.style.configure('TFrame', background='#2E2E2E')
        self.style.configure('TLabel', background='#2E2E2E', foreground='white')
        self.style.configure('TButton', background='#4A4A4A', foreground='white')
        self.style.map('TButton', background=[('active', '#6E6E6E')])
        self.style.configure('Horizontal.TScale', background='#2E2E2E')

        # Create Menu
        self.menu = tk.Menu(self, bg='#2E2E2E', fg='white')
        self.config(menu=self.menu)
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image, state=tk.DISABLED)
        self.file_menu = file_menu
        file_menu.add_separator()
        file_menu.add_command(label="About", command=self.show_about)
        file_menu.add_command(label="Exit", command=self.quit)

        # Main layout
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controls Frame
        self.controls_frame = ttk.Frame(main_frame, width=350)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.controls_frame.pack_propagate(False)

        # --- WIDGETS ---

        # Model
        ttk.Label(self.controls_frame, text="Model").pack(pady=(10,0))
        self.model_type = tk.StringVar(value='DPT_Large')
        model_combo = ttk.Combobox(self.controls_frame, textvariable=self.model_type, values=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'])
        model_combo.pack(pady=(0,10), fill=tk.X, padx=5)

        # Sliders
        self.blades = self._create_slider("Blades", 0, 20, 8)
        self.angle = self._create_slider("Angle", 0, 360, 0)
        self.max_radius = self._create_slider("Max Radius", 1, 100, 28)
        self.sharpness = self._create_slider("Sharpness", 0, 100, 12)
        self.band = self._create_slider("Band", 0.01, 1.0, 0.1, is_float=True)
        self.mask_feather = self._create_slider("Mask Feather", 0, 50, 0)
        self.layers = self._create_slider("Layers", 1, 20, 1)
        self.layer_blur_scale = self._create_slider("Layer Blur Scale", 0.1, 5.0, 1.0, is_float=True)

        # Checkboxes
        self.invert_depth = tk.BooleanVar()
        self.guided_mask = tk.BooleanVar()
        ttk.Checkbutton(self.controls_frame, text="Invert Depth", variable=self.invert_depth).pack(pady=5)
        ttk.Checkbutton(self.controls_frame, text="Guided Mask", variable=self.guided_mask).pack(pady=5)

        # Process Button
        self.process_button = ttk.Button(self.controls_frame, text="Process Image", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(pady=20)

        # Image Frame
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(self.image_frame, text="Load an image to start", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Button-1>", self.on_image_click)

    def open_image(self):
        filepath = filedialog.askopenfilename()
        if not filepath:
            return
        self.original_image = cv2.imread(filepath)
        self.display_image(self.original_image)
        self.file_menu.entryconfig("Save", state=tk.NORMAL)
        self.process_button.config(state=tk.NORMAL)

    def display_image(self, img_cv2):
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Resize image to fit the label
        w, h = self.image_frame.winfo_width(), self.image_frame.winfo_height()
        if w == 1 or h == 1: # Frame not rendered yet
            self.after(100, lambda: self.display_image(img_cv2))
            return

        img_pil.thumbnail((w - 20, h - 20))

        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.tk_image)

    def on_image_click(self, event):
        if self.original_image is None:
            return

        # Convert click coordinates to image coordinates
        x = event.x
        y = event.y

        label_w = self.image_label.winfo_width()
        label_h = self.image_label.winfo_height()

        img_w, img_h = self.tk_image.width(), self.tk_image.height()

        # Center the image in the label
        offset_x = (label_w - img_w) / 2
        offset_y = (label_h - img_h) / 2

        if offset_x < x < offset_x + img_w and offset_y < y < offset_y + img_h:
            self.focus_point = ((x - offset_x) / img_w, (y - offset_y) / img_h)

            # Draw a marker on the displayed image
            img_with_marker = self.original_image.copy()
            marker_pos = (int(self.focus_point[0] * self.original_image.shape[1]), int(self.focus_point[1] * self.original_image.shape[0]))
            cv2.circle(img_with_marker, marker_pos, 10, (0, 255, 0), 2)
            self.display_image(img_with_marker)

            print(f"Focus point set to: {self.focus_point}")

    def process_image(self):
        if self.original_image is None:
            return

        self.process_button.config(state=tk.DISABLED, text="Processing...")

        thread = threading.Thread(target=self._process_thread)
        thread.start()

    def _process_thread(self):
        try:
            # Initialize or update processor
            if self.processor is None or self.processor.model_type != self.model_type.get():
                self.processor = BokehProcessor(model_type=self.model_type.get())
                self.processor.generate_depth_map(self.original_image)

            self.processed_image = self.processor.process_image(
                self.original_image,
                self.focus_point,
                self.blades.get(),
                self.angle.get(),
                self.max_radius.get(),
                self.sharpness.get(),
                self.band.get(),
                self.mask_feather.get(),
                self.guided_mask.get(),
                self.layers.get(),
                self.layer_blur_scale.get(),
                self.invert_depth.get()
            )
            self.display_image(self.processed_image)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.process_button.config(state=tk.NORMAL, text="Process Image")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showinfo("Nothing to save", "Please process an image first.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if not filepath:
            return
        cv2.imwrite(filepath, self.processed_image)
        messagebox.showinfo("Saved", f"Image saved to {filepath}")

    def _create_slider(self, label, from_, to, default, is_float=False):
        frame = ttk.Frame(self.controls_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        var_type = tk.DoubleVar if is_float else tk.IntVar
        var = var_type(value=default)
        slider = ttk.Scale(frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL)
        slider.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        return var

    def show_about(self):
        messagebox.showinfo("About", "Zer0-Day Bokeh\n\nA cybernetic depth field adjuster.\n\nGitHub: https://github.com/your-repo")

if __name__ == "__main__":
    app = App()
    app.mainloop()
