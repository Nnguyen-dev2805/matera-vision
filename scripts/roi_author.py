import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

from PIL import Image, ImageTk


class ROIAuthorApp:
    def __init__(self, root, image_path: Path, output_path: Path):
        self.root = root
        self.root.title(f"ROI Authoring - {image_path.name}")
        self.output_path = output_path

        self.regions = []  # list of dicts: label, x, y, w, h

        # Load existing if any
        if self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    self.regions = json.load(f)
            except Exception as e:
                print(f"Failed to load existing {self.output_path}: {e}")

        # Load Image
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # UI Setup
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(fill=tk.X)

        self.save_btn = tk.Button(self.top_frame, text="Save (Ctrl+S)", command=self.save)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_btn = tk.Button(self.top_frame, text="Clear All", command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.info_label = tk.Label(
            self.top_frame, text="Drag to draw box. Label format: 'anchor:name' or 'roi:Q1/a'"
        )
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Canvas with scrollbars
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame, scrollregion=(0, 0, self.image.width, self.image.height)
        )
        self.hbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Drawing state
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        # Events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Control-s>", lambda e: self.save())

        self.redraw_regions()

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2
        )

    def on_drag(self, event):
        if self.current_rect:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        if not self.current_rect:
            return

        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        w = int(x2 - x1)
        h = int(y2 - y1)

        if w < 5 or h < 5:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            return

        label = simpledialog.askstring(
            "Input", "Enter label (e.g., 'anchor:top_left_qr' or 'roi:Q1/a'):", parent=self.root
        )

        self.canvas.delete(self.current_rect)
        self.current_rect = None

        if label:
            self.regions.append(
                {"label": label, "x": int(x1), "y": int(y1), "w": w, "h": h}
            )
            self.redraw_regions()

    def clear_all(self):
        if messagebox.askyesno("Confirm", "Clear all drawn regions?"):
            # mark history if needed, but lets just clear
            self.regions.append({"_cleared": True})
            self.regions = []
            self.redraw_regions()

    def redraw_regions(self):
        self.canvas.delete("region")
        for r in self.regions:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            color = "blue" if r["label"].startswith("anchor") else "red"
            self.canvas.create_rectangle(
                x, y, x + w, y + h, outline=color, width=2, tags="region"
            )
            self.canvas.create_text(
                x,
                y - 10,
                text=r["label"],
                fill=color,
                anchor="w",
                tags="region",
                font=("Arial", 12, "bold"),
            )

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.regions, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Saved", f"Saved {len(self.regions)} regions to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="ROI Authoring Tool")
    parser.add_argument(
        "--image", required=True, type=Path, help="Path to the template image (e.g., page_1.png)"
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to the output JSON file")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image {args.image} not found.")
        return 1

    root = tk.Tk()
    _app = ROIAuthorApp(root, args.image, args.output)

    # Maximize window
    try:
        root.state("zoomed")
    except Exception:
        pass

    root.mainloop()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
