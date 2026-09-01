import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk


class BoxSelectorApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Chọn Tọa Độ Q14 (Đã Thu Nhỏ 50%)")

        # Setup UI
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, cursor="cross")
        self.vbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.hbar = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.config(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill=tk.X, pady=5)

        self.btn_clear = tk.Button(self.btn_frame, text="Xóa Box Vừa Vẽ", command=self.clear_last)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.btn_save = tk.Button(
            self.btn_frame, text="Lưu & Thoát", command=self.save_and_exit, bg="green", fg="white"
        )
        self.btn_save.pack(side=tk.RIGHT, padx=10)

        # Load image
        self.original_image = Image.open(image_path).convert("RGB")
        self.scale = 0.4  # Scale down to 40% to fit screen better

        new_width = int(self.original_image.width * self.scale)
        new_height = int(self.original_image.height * self.scale)

        self.display_image = self.original_image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Variables for drawing
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.boxes_original = []  # List of dicts {x, y, w, h} in original scale
        self.rect_ids = []

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Scroll down to bottom since Q14 is at the bottom
        self.canvas.yview_moveto(0.8)

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2
        )

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        w = x2 - x1
        h = y2 - y1

        if w > 10 and h > 10:
            # Scale coordinates back to original size
            orig_x = int(x1 / self.scale)
            orig_y = int(y1 / self.scale)
            orig_w = int(w / self.scale)
            orig_h = int(h / self.scale)

            box = {"x": orig_x, "y": orig_y, "w": orig_w, "h": orig_h}
            self.boxes_original.append(box)
            self.rect_ids.append(self.current_rect)
            print(f"Đã vẽ box (original coords): {box}")
        else:
            self.canvas.delete(self.current_rect)

    def clear_last(self):
        if self.rect_ids:
            rect = self.rect_ids.pop()
            self.canvas.delete(rect)
            self.boxes_original.pop()

    def save_and_exit(self):
        if not self.boxes_original:
            messagebox.showwarning("Cảnh báo", "Bạn chưa vẽ box nào!")
            return

        out_file = Path("q14_custom_coords.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"q14_boxes": self.boxes_original}, f, indent=2)

        messagebox.showinfo(
            "Thành công", f"Đã lưu {len(self.boxes_original)} boxes vào {out_file.absolute()}"
        )
        self.root.destroy()


def main():
    ref_path = Path("data/references/cleaner_sessions/smoke/reference_original.png")
    if not ref_path.exists():
        print(f"Không tìm thấy ảnh {ref_path}")
        return 1

    root = tk.Tk()
    root.geometry("1000x800")
    _ = BoxSelectorApp(root, ref_path)
    root.mainloop()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
