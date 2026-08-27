import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


class CropApp:
    def __init__(self, root, img_path, out_dir):
        self.root = root
        self.root.title("Công cụ Cắt Ảnh (Matera Vision)")

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.img_path = img_path

        # Load ảnh gốc
        try:
            self.original_img = Image.open(img_path)
        except Exception as e:
            print(f"Lỗi tải ảnh: {e}")
            self.root.quit()
            return

        self.tk_img = ImageTk.PhotoImage(self.original_img)

        # Khung chứa canvas và thanh cuộn
        frame = tk.Frame(root)
        frame.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(frame, cursor="cross")

        # Thanh cuộn dọc & ngang cho ảnh bự
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.config(scrollregion=(0, 0, self.original_img.width, self.original_img.height))

        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

        # Bắt sự kiện chuột và phím
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<Return>", self.save_crop)
        self.root.bind("<space>", self.save_crop)
        self.root.bind("<Escape>", lambda e: self.root.quit())

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.count = 1

        print("===" * 15)
        print("HƯỚNG DẪN SỬ DỤNG:")
        print("1. Kéo thanh cuộn để xem toàn bộ ảnh.")
        print("2. Kéo thả chuột để vẽ khung ĐỎ vùng cần cắt.")
        print("3. Nhấn phím ENTER (hoặc SPACE) để LƯU vùng đang chọn.")
        print("4. Kéo vùng chọn mới sẽ tự động bỏ vùng chọn cũ chưa lưu.")
        print("5. Nhấn phím ESC để thoát.")
        print("===" * 15)

    def on_press(self, event):
        # Lấy tọa độ thực tế trên ảnh có scroll
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=3
        )

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def save_crop(self, event):
        if not self.rect:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        # Chỉnh lại cho đúng left, top, right, bottom (trường hợp kéo chuột ngược)
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        if right - left <= 0 or bottom - top <= 0:
            return

        # Cắt từ ảnh gốc PIL
        crop_img = self.original_img.crop((left, top, right, bottom))
        out_path = self.out_dir / f"crop_cau_{self.count}.png"
        crop_img.save(out_path)
        print(f"Đã lưu thành công: {out_path}")

        # Để nguyên khung đỏ trên màn hình như một dấu hiệu đã xử lý xong
        # Và reset rect để lấn kéo tiếp theo tạo khung mới
        self.rect = None
        self.count += 1


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    # Đảm bảo đường dẫn ảnh chuẩn
    img_path = Path("data/pages/page_1.png")
    out_dir = Path("data/debug/manual_crops")

    app = CropApp(root, img_path, out_dir)
    root.mainloop()
