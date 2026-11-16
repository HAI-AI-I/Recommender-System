import argparse
import queue
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from phase4_recommender import DATA_DIR, MovieRecommender


class MovieDemoApp:
    """Giao diện mỏng; toàn bộ tính toán nằm trong MovieRecommender."""

    def __init__(self, root, data_dir=DATA_DIR):
        self.root = root
        self.data_dir = data_dir
        self.engine = None
        self.results = pd.DataFrame()
        self.messages = queue.Queue()
        self.busy = False
        self.user = tk.StringVar()
        self.top_k = tk.StringVar(value="10")
        self.genre = tk.StringVar(value="Tất cả")
        self.mode = tk.StringVar(value="Chỉ CF")
        self.status = tk.StringVar(value="Đang khởi động...")
        self.summary = tk.StringVar(value="Nạp mô hình và lịch sử một lần để gợi ý nhanh.")
        self.model_status = tk.StringVar()
        self.controls = []
        self._build_window()
        self._poll_job = self.root.after(100, self._poll_messages)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._reload()

    def _build_window(self):
        self.root.title("Gợi ý phim MovieLens_32M")
        self.root.geometry("1200x760")
        self.root.minsize(1000, 620)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", padding=(16, 9), foreground="white", background="#2563eb")
        body = ttk.Frame(self.root, padding=24)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Khám phá phim dành cho bạn", style="Title.TLabel").pack(anchor="w")
        ttk.Label(body, text="300 ứng viên từ lọc cộng tác → xếp hạng LightGBM → danh sách phim chưa xem").pack(anchor="w", pady=(4, 14))
        ttk.Label(body, textvariable=self.model_status).pack(anchor="w", pady=(0, 14))

        controls = ttk.Frame(body)
        controls.pack(fill="x")
        user_group = ttk.Frame(controls)
        user_group.pack(side="left", padx=(0, 14))
        ttk.Label(user_group, text="ID người dùng").pack(anchor="w")
        entry = ttk.Entry(user_group, textvariable=self.user, width=13)
        entry.pack(side="left", pady=5)
        random_button = ttk.Button(user_group, text="Ngẫu nhiên", command=self._random_user)
        random_button.pack(side="left", padx=(6, 0))
        self.controls.extend([(entry, "normal"), (random_button, "normal")])
        for label, variable, values, width in [
            ("Số phim", self.top_k, [5, 10, 20, 50], 7),
            ("Thể loại", self.genre, ["Tất cả"], 18),
            ("Cách xếp hạng", self.mode, ["Chỉ CF"], 19),
        ]:
            frame = ttk.Frame(controls)
            frame.pack(side="left", padx=(0, 14))
            ttk.Label(frame, text=label).pack(anchor="w")
            combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=width)
            combo.pack(pady=5)
            self.controls.append((combo, "readonly"))
            if variable is self.genre:
                self.genre_combo = combo
            if variable is self.mode:
                self.mode_combo = combo
        button = ttk.Button(controls, text="Gợi ý phim", command=self._recommend, style="Accent.TButton")
        button.pack(side="right", pady=(15, 0))
        self.controls.append((button, "normal"))

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(12, 14))
        self.reload_button = ttk.Button(actions, text="Nạp lại kết quả giai đoạn 3", command=self._reload)
        self.reload_button.pack(side="left", padx=(0, 10))
        self.controls.append((self.reload_button, "normal"))
        for label, callback in [("Xuất danh sách CSV", self._export),
                                ("Đánh giá trên 100 người dùng test", self._evaluate)]:
            button = ttk.Button(actions, text=label, command=callback)
            button.pack(side="left", padx=(0, 10))
            self.controls.append((button, "normal"))
        ttk.Label(body, textvariable=self.summary).pack(anchor="w", pady=(0, 10))

        self.tabs = ttk.Notebook(body)
        self.tabs.pack(fill="both", expand=True)
        self.recommendations = self._table("Phim gợi ý", [
            ("rank", "Hạng", 45), ("movieId", "ID phim", 70), ("clean_title", "Tên phim", 300),
            ("release_year", "Năm", 65), ("genres", "Thể loại", 200),
            ("predicted_like_prob", "Xác suất thích", 110), ("cf_score", "Điểm CF", 85),
            ("movie_avg_rating", "Điểm TB train", 105),
        ])
        self.candidates = self._table("Ứng viên vòng 1 (Top 300)", [
            ("cf_rank", "Hạng vòng 1", 95), ("movieId", "ID phim", 80),
            ("clean_title", "Tên phim", 400), ("release_year", "Năm", 75),
            ("genres", "Thể loại", 260), ("cf_score", "Điểm CF", 100),
        ])
        self.history = self._table("Lịch sử đánh giá", [
            ("movieId", "ID phim", 80), ("clean_title", "Tên phim", 430),
            ("release_year", "Năm", 90), ("genres", "Thể loại", 260), ("rating", "Điểm đã chấm", 120),
        ])
        self.evaluation_tab = ttk.Frame(self.tabs, padding=24)
        self.tabs.add(self.evaluation_tab, text="Đánh giá mô hình")
        self.evaluation_text = tk.StringVar(value="Chọn cách xếp hạng và nhấn Đánh giá để xem chất lượng trên tập test.\n"
                                                "Chỉ tính phim thích chưa xem trong lịch sử train; lấy tối đa 100 người dùng hợp lệ.")
        ttk.Label(self.evaluation_tab, textvariable=self.evaluation_text, justify="left", font=("Segoe UI", 12)).pack(anchor="nw")
        self.progress_bar = ttk.Progressbar(body, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(14, 5))
        ttk.Label(body, textvariable=self.status, wraplength=1100).pack(anchor="w")
        self.root.bind("<Return>", lambda event: self._recommend() if not self.busy and self.engine else None)

    def _table(self, title, columns):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text=title)
        table = ttk.Treeview(frame, columns=[key for key, _, _ in columns], show="headings", selectmode="browse")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        for key, label, width in columns:
            table.heading(key, text=label)
            table.column(key, width=width, minwidth=45, anchor="w", stretch=key == "clean_title")
        return table

    def _progress(self, text):
        self.messages.put(("progress", text))

    def _run(self, work, finished):
        """Chạy việc nặng ngoài luồng giao diện; chỉ cập nhật Tkinter ở luồng chính."""
        if self.busy:
            return
        self.busy = True
        for widget, _ in self.controls:
            widget.configure(state="disabled")
        self.progress_bar.start(12)

        def worker():
            try:
                self.messages.put(("done", finished, work()))
            except Exception as error:
                self.messages.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_messages(self):
        while not self.messages.empty():
            message = self.messages.get_nowait()
            if message[0] == "progress":
                self.status.set(message[1])
                continue
            self.busy = False
            self.progress_bar.stop()
            for widget, state in self.controls:
                enabled = self.engine is not None or message[0] == "done" or widget is self.reload_button
                widget.configure(state=state if enabled else "disabled")
            if message[0] == "error":
                self.status.set(f"Lỗi: {message[1]}")
                messagebox.showerror("Không thể hoàn tất", message[1], parent=self.root)
            else:
                message[1](message[2])
        if self._poll_job is not None:
            self._poll_job = self.root.after(100, self._poll_messages)

    def _close(self):
        """Hủy lịch cập nhật trước khi đóng để không còn callback gọi cửa sổ đã hủy."""
        if self._poll_job is None:
            return
        self.root.after_cancel(self._poll_job)
        self._poll_job = None
        self.root.destroy()

    def _ready(self, engine):
        self.engine = engine
        self.results = pd.DataFrame()
        for table in (self.recommendations, self.candidates, self.history):
            table.delete(*table.get_children())
        self.user.set(str(engine.user_ids[0]))
        self.genre.set("Tất cả")
        self.genre_combo.configure(values=["Tất cả"] + engine.genres)
        self._update_model_status()
        self.summary.set("Chọn người dùng để xem 300 ứng viên vòng 1 và kết quả xếp hạng cuối cùng.")
        self.evaluation_text.set("Chọn cách xếp hạng và nhấn Đánh giá để xem chất lượng trên tập test.")
        self.status.set("Đã sẵn sàng. Chọn người dùng và nhấn Gợi ý phim.")

    def _reload(self):
        """Nạp lại các file sau khi chạy notebook, không huấn luyện trong giao diện."""
        self.status.set("Đang nạp kết quả đã xuất từ notebook giai đoạn 3...")
        self._run(lambda: MovieRecommender(self.data_dir, self._progress), self._ready)

    def _update_model_status(self):
        ready = self.engine.ranker is not None
        self.mode_combo.configure(values=["Chỉ CF", "CF + LightGBM"] if ready else ["Chỉ CF"])
        self.mode.set("CF + LightGBM" if ready else "Chỉ CF")
        self.model_status.set(f"Đã nạp {len(self.engine.users):,} người dùng • {len(self.engine.movies):,} phim • "
                              + (f"CF {self.engine.user_vectors.shape[1]} chiều • LightGBM {len(self.engine.feature_columns)} đặc trưng"
                                 if ready else "Chưa có LightGBM; hãy chạy notebook giai đoạn 3"))

    def _random_user(self):
        self.user.set(str(int(random.choice(self.engine.user_ids))))
        self._recommend()

    def _recommend(self):
        try:
            user_id, top_k = int(self.user.get()), int(self.top_k.get())
            if user_id < 1 or not 1 <= top_k <= 50:
                raise ValueError
        except ValueError:
            messagebox.showerror("Dữ liệu chưa hợp lệ", "ID phải là số nguyên dương; số phim phải từ 1 đến 50.", parent=self.root)
            return
        genre = None if self.genre.get() == "Tất cả" else self.genre.get()
        hybrid = self.mode.get() == "CF + LightGBM"
        self.status.set("Đang lọc ứng viên và xếp hạng phim...")

        def work():
            candidates = self.engine.get_candidates(user_id, genre)
            result = self.engine.rank_candidates(user_id, candidates, top_k, hybrid)
            return result, self.engine.get_history(user_id), candidates

        def show(data):
            self.results, history, candidates = data
            self._fill(self.recommendations, self.results)
            self._fill(self.candidates, candidates)
            self._fill(self.history, history)
            position = self.engine.users.get_indexer([user_id])[0]
            if position >= 0:
                count = self.engine.history.indptr[position + 1] - self.engine.history.indptr[position]
                description = f"Người dùng {user_id} • {count:,} phim đã đánh giá • Lịch sử hiển thị tối đa 100 phim, ưu tiên điểm cao"
            else:
                description = f"Người dùng mới {user_id} • Bắt đầu từ phim phổ biến trong tập train"
            self.summary.set(description)
            self.status.set(f"Đã lọc {len(candidates)} ứng viên vòng 1 → trả {len(self.results)} phim chưa xem "
                            f"sau xếp hạng {'LightGBM' if hybrid else 'CF'}." if len(self.results)
                            else "Không còn phim phù hợp với bộ lọc. Hãy thử thể loại khác.")
            self.tabs.select(0)

        self._run(work, show)

    @staticmethod
    def _fill(table, frame):
        table.delete(*table.get_children())
        for row in frame.to_dict("records"):
            values = []
            for column in table["columns"]:
                value = row.get(column, "")
                if pd.isna(value):
                    value = "—"
                elif column == "predicted_like_prob":
                    value = f"{value:.1%}"
                elif column in {"cf_score", "movie_avg_rating", "rating"}:
                    value = f"{value:.2f}"
                values.append(value)
            table.insert("", "end", values=values)

    def _export(self):
        if self.results.empty:
            messagebox.showinfo("Chưa có kết quả", "Hãy tạo danh sách gợi ý trước khi xuất CSV.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Lưu danh sách phim gợi ý",
                                           defaultextension=".csv", initialfile=f"goi_y_phim_{self.results.userId.iloc[0]}.csv",
                                           filetypes=[("Dữ liệu CSV", "*.csv")])
        if path:
            try:
                self.results.to_csv(path, index=False, encoding="utf-8-sig")
                self.status.set(f"Đã lưu danh sách CSV: {path}")
            except OSError as error:
                messagebox.showerror("Không lưu được file", str(error), parent=self.root)

    def _evaluate(self):
        hybrid = self.mode.get() == "CF + LightGBM"
        top_k = int(self.top_k.get())
        self.status.set("Đang đánh giá trên tập test; giao diện vẫn có thể di chuyển...")

        def show(metrics):
            mode = "CF + LightGBM" if hybrid else "Chỉ CF"
            self.evaluation_text.set(
                f"Chế độ: {mode}\nSố người dùng được đánh giá: {metrics['users']}\nSố phim gợi ý: {metrics['k']}\n\n"
                f"Độ thu hồi (Recall): {metrics['recall']:.2%}\n"
                f"Độ chính xác (Precision): {metrics['precision']:.2%}\n"
                f"Tỷ lệ gợi ý trúng (Hit Rate): {metrics['hit_rate']:.2%}\n"
                f"Chất lượng thứ hạng (NDCG): {metrics['ndcg']:.2%}\n\n"
                "Các chỉ số được lấy trung bình theo người dùng.\n"
                "Đặc trưng và mô hình được nạp từ notebook giai đoạn 3.\n"
                "Lưu ý: các bảng thống kê trong notebook được tính trên toàn bộ ratings."
            )
            self.tabs.select(self.evaluation_tab)
            self.status.set("Đã hoàn tất đánh giá mô hình.")

        self._run(lambda: self.engine.evaluate(100, top_k, hybrid), show)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Thư mục dữ liệu và mô hình")
    args = parser.parse_args()
    root = tk.Tk()
    MovieDemoApp(root, args.data_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
