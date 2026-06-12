import csv
import json
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
        style = ttk.Style(self)
        style.configure("Run.TButton", font=("Arial", 10, "bold"))

        self.build_ui()



    def build_ui(self):
        title = ttk.Label(
            self,
            text="Dirtify Workbench Prototype",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=(15, 2))

        subtitle = ttk.Label(
            self,
            text="Dataset management • Analysis configuration • Results workflow",
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 8))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=12, pady=(0, 10))

        frame = ttk.LabelFrame(self, text="Workbench workflow", padding=12)
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        frame.columnconfigure(0, minsize=130)

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

        ttk.Label(frame, text="Exploratory Analysis:").grid(row=5, column=0, sticky="nw", pady=5)

        analysis_tools_frame = ttk.Frame(frame)
        analysis_tools_frame.grid(row=5, column=1, sticky="w", pady=5)

        ttk.Button(
            analysis_tools_frame,
            text="Show Distributions",
            command=self.show_distributions
        ).grid(row=0, column=0, padx=(0, 10))

        ttk.Button(
            analysis_tools_frame,
            text="Show Correlations",
            command=self.show_correlations
        ).grid(row=0, column=1)


        ttk.Button(
            frame,
            text="Run Analysis",
            command=self.run_analysis,
            style="Run.TButton"
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 10))


        ttk.Label(frame, text="Results View:").grid(row=7, column=0, sticky="nw", pady=5)

        results_tools_frame = ttk.Frame(frame)
        results_tools_frame.grid(row=7, column=1, sticky="w", pady=5)

        ttk.Button(
            results_tools_frame,
            text="Show Results Table",
            command=self.show_results_table
        ).grid(row=0, column=0, padx=(0, 10))


        ttk.Button(
            results_tools_frame,
            text="Show Graph Placeholder",
            command=self.show_graph_placeholder
        ).grid(row=0, column=1)


        ttk.Button(
            results_tools_frame,
            text="Save Project",
            command=self.save_project
        ).grid(row=0, column=2, padx=(10, 0))


        ttk.Button(
            results_tools_frame,
            text="Save Results",
            command=self.save_results
        ).grid(row=0, column=3, padx=(10, 0))


        ttk.Button(
            results_tools_frame,
            text="Load Project",
            command=self.load_project
        ).grid(row=0, column=4, padx=(10, 0))


        log_frame = ttk.LabelFrame(frame, text="Log / Results")
        log_frame.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=10)

        self.output_box = tk.Text(
            log_frame,
            height=8,
            width=60,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#C8C8C8",
            highlightcolor="#C8C8C8"
        )

        self.output_box.pack(fill="both", expand=True, padx=5, pady=5)


        preview_outer_frame = ttk.LabelFrame(frame, text="Dataset Preview")
        preview_outer_frame.grid(row=9, column=0, columnspan=2, sticky="nsew", pady=10)

        preview_border = tk.Frame(
            preview_outer_frame,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#C8C8C8",
            highlightcolor="#C8C8C8"
        )
        preview_border.pack(fill="both", expand=True, padx=5, pady=5)

        preview_frame = ttk.Frame(preview_border)
        preview_frame.pack(fill="both", expand=True, padx=0, pady=0)


        self.preview_tree = ttk.Treeview(
            preview_frame,
            show="headings",
            height=12
        )
        self.preview_tree.grid(row=0, column=0, sticky="nsew")

        

        scrollbar_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_tree.xview)
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        self.preview_tree.configure(
            xscrollcommand=scrollbar_x.set
        )

        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        ttk.Label(
            frame,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w"
        ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)
        frame.rowconfigure(9, weight=1)


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


    def show_distributions(self):
        self.log("Exploratory analysis: distributions requested.")
        self.status_var.set("Showing distributions")

    def show_correlations(self):
        self.log("Exploratory analysis: correlations requested.")
        self.status_var.set("Showing correlations")

    def show_results_table(self):
        self.log("Results view: table requested.")
        self.status_var.set("Showing results table")


    def show_graph_placeholder(self):
        self.log("Results view: graph placeholder requested.")
        self.status_var.set("Showing graph placeholder")

    def save_project(self):
        project_data = {
            "dataset_path": self.dataset_path.get(),
            "testset_path": self.testset_path.get(),
            "target_variable": self.target_var.get(),
            "analysis_strategy": self.strategy_var.get(),
            "custom_rules": self.custom_rules_box.get("1.0", tk.END).strip()
        }

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if path:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(project_data, file, indent=4)

            self.log(f"Project saved: {path}")
            self.status_var.set("Project saved")


    def save_results(self):
        results_text = self.output_box.get("1.0", tk.END).strip()

        if not results_text:
            self.log("No results available to save.")
            self.status_var.set("No results to save")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if path:
            with open(path, "w", encoding="utf-8") as file:
                file.write(results_text)

            self.log(f"Results saved: {path}")
            self.status_var.set("Results saved")
            

    def load_project(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if path:
            with open(path, "r", encoding="utf-8") as file:
                project_data = json.load(file)

            dataset_path = project_data.get("dataset_path", "No dataset selected")
            testset_path = project_data.get("testset_path", "No testset selected")
            target_variable = project_data.get("target_variable", "")
            analysis_strategy = project_data.get("analysis_strategy", "custom_rules")
            custom_rules = project_data.get("custom_rules", "")

            self.dataset_path.set(dataset_path)
            self.testset_path.set(testset_path)
            self.strategy_var.set(analysis_strategy)

            self.custom_rules_box.delete("1.0", tk.END)
            self.custom_rules_box.insert(tk.END, custom_rules)

            if dataset_path != "No dataset selected":
                self.ui_logic.open_dataset(dataset_path)
                self.load_columns_from_csv(dataset_path)

            if target_variable:
                self.target_var.set(target_variable)

            if testset_path != "No testset selected":
                self.ui_logic.open_testset(testset_path)

            self.log(f"Project loaded: {path}")
            self.status_var.set("Project loaded")
    

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