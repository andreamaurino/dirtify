import tkinter as tk
from tkinter import ttk, filedialog
from ui_layer import DataAnalysisUI


class DirtifyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dirtify UI Prototype")
        self.geometry("700x500")

        self.ui_logic = DataAnalysisUI()

        self.dataset_path = tk.StringVar(value="No dataset selected")
        self.target_var = tk.StringVar()
        self.strategy_var = tk.StringVar(value="custom_rules")

        self.build_ui()

    def build_ui(self):
        title = ttk.Label(self, text="Dirtify Workbench Prototype", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Button(frame, text="Open Dataset", command=self.open_dataset).grid(row=0, column=0, sticky="ew", pady=5)
        ttk.Label(frame, textvariable=self.dataset_path).grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(frame, text="Target Variable:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.target_var).grid(row=1, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Analysis Strategy:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(
            frame,
            textvariable=self.strategy_var,
            values=["custom_rules", "exploratory_analysis", "standard_analysis"],
            state="readonly"
        ).grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Button(frame, text="Run Analysis", command=self.run_analysis).grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(frame, text="Log / Results:").grid(row=4, column=0, sticky="nw", pady=5)
        self.output_box = tk.Text(frame, height=12, width=60)
        self.output_box.grid(row=4, column=1, sticky="nsew", pady=5)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

    def open_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.dataset_path.set(path)
            self.ui_logic.open_dataset(path)
            self.log(f"Dataset selected: {path}")

    def run_analysis(self):
        target = self.target_var.get().strip()
        strategy = self.strategy_var.get().strip()

        if self.dataset_path.get() == "No dataset selected":
            self.log("Please select a dataset first.")
            return

        if not target:
            self.log("Please insert a target variable.")
            return

        self.ui_logic.select_target_variable(target)
        self.ui_logic.select_analysis_strategy(strategy)
        self.ui_logic.run_analysis("json/example_config.json")

        self.log("Analysis started from UI prototype.")
        self.log(f"Target: {target}")
        self.log(f"Strategy: {strategy}")
        self.log("This is a visible UI prototype. Backend integration will come later.")

    def log(self, message):
        self.output_box.insert(tk.END, message + "\n")
        self.output_box.see(tk.END)


if __name__ == "__main__":
    app = DirtifyApp()
    app.mainloop()