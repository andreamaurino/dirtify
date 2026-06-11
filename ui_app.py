import csv
import tkinter as tk
from tkinter import ttk, filedialog
from ui_layer import DataAnalysisUI


class DirtifyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dirtify UI Prototype")
        self.geometry("900x650")

        self.ui_logic = DataAnalysisUI()

        self.dataset_path = tk.StringVar(value="No dataset selected")
        self.testset_path = tk.StringVar(value="No testset selected")
        self.target_var = tk.StringVar()
        self.strategy_var = tk.StringVar(value="custom_rules")
        self.status_var = tk.StringVar(value="Ready")

        self.build_ui()

    def build_ui(self):
        title = ttk.Label(self, text="Dirtify Workbench Prototype", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Button(frame, text="Open Dataset", command=self.open_dataset).grid(row=0, column=0, sticky="ew", pady=5)
        ttk.Label(frame, textvariable=self.dataset_path, wraplength=500).grid(row=0, column=1, sticky="w", padx=10)

        ttk.Button(frame, text="Open Testset", command=self.open_testset).grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Label(frame, textvariable=self.testset_path, wraplength=500).grid(row=1, column=1, sticky="w", padx=10)


        ttk.Label(frame, text="Target Variable:").grid(row=2, column=0, sticky="w", pady=5)
        self.target_combo = ttk.Combobox(
            frame,
            textvariable=self.target_var,
            state="readonly",
            values=[]
        )
        self.target_combo.grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Analysis Strategy:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(
            frame,
            textvariable=self.strategy_var,
            values=["custom_rules", "exploratory_analysis", "standard_analysis"],
            state="readonly"
        ).grid(row=3, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Custom Rules:").grid(row=4, column=0, sticky="nw", pady=5)
        self.custom_rules_box = tk.Text(frame, height=4, width=60)
        self.custom_rules_box.grid(row=4, column=1, sticky="ew", pady=5)
        self.custom_rules_box.insert(
            tk.END,
            "Example: IF age > 40 THEN apply custom modification\n"
        )

        ttk.Button(frame, text="Run Analysis", command=self.run_analysis).grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(frame, text="Log / Results:").grid(row=6, column=0, sticky="nw", pady=5)
        self.output_box = tk.Text(frame, height=8, width=60)
        self.output_box.grid(row=6, column=1, sticky="nsew", pady=5)

        ttk.Label(frame, text="Dataset Preview:").grid(row=7, column=0, sticky="nw", pady=(10, 5))
        preview_frame = ttk.Frame(frame)
        preview_frame.grid(row=7, column=1, sticky="nsew", pady=(10, 5))

        self.preview_tree = ttk.Treeview(preview_frame, show="headings", height=12)
        self.preview_tree.grid(row=0, column=0, sticky="nsew")

        scrollbar_y = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        scrollbar_y.grid(row=0, column=1, sticky="ns")

        scrollbar_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_tree.xview)
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        self.preview_tree.configure(
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )

        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        ttk.Label(
            frame,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w"
        ).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)
        frame.rowconfigure(7, weight=1)


    def open_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.dataset_path.set(path)
            self.status_var.set("Dataset loaded")
            self.ui_logic.open_dataset(path)
            self.log(f"Dataset selected: {path}")
            self.load_columns_from_csv(path)
            


    def open_testset(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
           self.testset_path.set(path)
           self.status_var.set("Testset loaded")
           self.ui_logic.open_testset(path)
           self.log(f"Testset selected: {path}")

    

    def load_columns_from_csv(self, path):
        try:
            with open(path, newline="", encoding="utf-8", errors="ignore") as file:
                reader = csv.reader(file)
                rows = list(reader)

            if not rows:
                self.log("The selected dataset is empty.")
                return

            headers = rows[0]
            preview_rows = rows[1:11]

            # update target dropdown
            self.target_combo["values"] = headers

            if headers:
                self.target_var.set(headers[0])

            # clear old preview
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)

            self.preview_tree["columns"] = headers
            self.preview_tree["show"] = "headings"

            for col in headers:
                self.preview_tree.heading(col, text=col)
                self.preview_tree.column(col, width=120, anchor="center")

            for row in preview_rows:
                padded_row = row + [""] * (len(headers) - len(row))
                self.preview_tree.insert("", tk.END, values=padded_row[:len(headers)])

            self.log(f"Detected columns: {', '.join(headers)}")
            self.log(f"Preview loaded: showing first {len(preview_rows)} rows.")

        except Exception as e:
            self.log(f"Error while reading dataset: {e}")



    def run_analysis(self):
        target = self.target_var.get().strip()
        strategy = self.strategy_var.get().strip()
        custom_rules = self.custom_rules_box.get("1.0", tk.END).strip()

        if self.dataset_path.get() == "No dataset selected":
            self.log("Please select a dataset first.")
            return

        if not target:
            self.log("Please select a target variable.")
            return

        self.ui_logic.select_target_variable(target)
        self.ui_logic.select_analysis_strategy(strategy)
        self.ui_logic.run_analysis("json/example_config.json")
        self.status_var.set("Analysis completed")

        self.log("Analysis started from UI prototype.")
        self.log(f"Target: {target}")
        self.log(f"Strategy: {strategy}")
        if custom_rules:
            self.log("Custom rules:")
            self.log(custom_rules)
        else:
            self.log("No custom rules provided.")

        self.log("This is a visible UI prototype. Backend integration will come later.")



    def log(self, message):
        self.output_box.insert(tk.END, message + "\n")
        self.output_box.see(tk.END)


if __name__ == "__main__":
    app = DirtifyApp()
    app.mainloop()