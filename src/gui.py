#!/usr/bin/env python3
"""
dTAO Price Predictor GUI - Tkinter-based graphical interface
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import csv
import os
import logging
import threading
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from src.data_fetcher import DataFetcher
from src.price_predictor import PricePredictor
from src.comparison_analyzer import ComparisonAnalyzer
from src.standalone import MockDataFetcher, MockPricePredictor, MockComparisonAnalyzer

logger = logging.getLogger(__name__)


class I18n:
    """Internationalization helper that loads JSON translation files."""

    def __init__(self, locale: str = "en_US"):
        self.locale = locale
        self.translations: Dict[str, str] = {}
        self._load(locale)

    # -----------------------------------------------------------------
    def _load(self, locale: str):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "i18n")
        path = os.path.join(base_dir, f"{locale}.json")
        if not os.path.exists(path):
            path = os.path.join(base_dir, "en_US.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.translations = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load i18n file {path}: {e}")
            self.translations = {}

    def get_text(self, key: str, default: str = "") -> str:
        return self.translations.get(key, default or key)

    def set_locale(self, locale: str):
        self.locale = locale
        self._load(locale)

    @staticmethod
    def available_locales() -> List[str]:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "i18n")
        locales = []
        if os.path.isdir(base_dir):
            for fname in sorted(os.listdir(base_dir)):
                if fname.endswith(".json"):
                    locales.append(fname.replace(".json", ""))
        return locales


class DTAOPredictorGUI:
    """Main GUI application for dTAO Price Predictor."""

    def __init__(self, root: tk.Tk, use_mock: bool = False):
        self.root = root
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # Internationalisation
        self.i18n = I18n("en_US")

        # Backend services (real by default, mock only when requested)
        self.backend_mode = "mock"
        self._init_backend(use_mock=use_mock or os.getenv("DTAO_USE_MOCK", "0") == "1")

        # Subnet cache
        self.subnets: List[Dict[str, Any]] = []

        # Last prediction result (for saving chart)
        self._last_prediction = None
        self._last_recommendations: List[Dict[str, Any]] = []
        self._last_mining_recommendations: List[Dict[str, Any]] = []

        self._build_ui()
        self._set_status(f"{self.i18n.get_text('status_ready', 'Ready')} ({self.backend_mode} backend)")

    def _init_backend(self, use_mock: bool = False):
        if use_mock:
            self._init_mock_backend()
            return

        try:
            self.data_fetcher = DataFetcher()
            self.price_predictor = PricePredictor(self.data_fetcher)
            self.comparison_analyzer = ComparisonAnalyzer(self.data_fetcher, self.price_predictor)
            self.backend_mode = "real"
            logger.info("Using real backend services")
        except Exception as e:
            logger.exception("Failed to initialize real backend, falling back to mock backend: %s", e)
            self._init_mock_backend()

    def _init_mock_backend(self):
        self.data_fetcher = MockDataFetcher()
        self.price_predictor = MockPricePredictor(self.data_fetcher)
        self.comparison_analyzer = MockComparisonAnalyzer(self.data_fetcher, self.price_predictor)
        self.backend_mode = "mock"
        logger.info("Using mock backend services")

    # ======================= UI construction ==========================

    def _build_ui(self):
        # Top bar: language selector
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(top_frame, text=self.i18n.get_text("language", "Language") + ":").pack(side=tk.LEFT, padx=5)
        self.locale_var = tk.StringVar(value=self.i18n.locale)
        locale_cb = ttk.Combobox(top_frame, textvariable=self.locale_var,
                                 values=I18n.available_locales(), state="readonly", width=10)
        locale_cb.pack(side=tk.LEFT, padx=5)
        locale_cb.bind("<<ComboboxSelected>>", self._on_locale_changed)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._create_subnet_list_tab()
        self._create_subnet_info_tab()
        self._create_price_prediction_tab()
        self._create_subnet_comparison_tab()
        self._create_investment_tab()
        self._create_mining_tab()

        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)

    # -------------------- Tab 1: Subnet List --------------------------

    def _create_subnet_list_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.i18n.get_text("tab_subnet_list", "Subnet List"))

        ctrl = ttk.LabelFrame(tab, text=self.i18n.get_text("options", "Options"))
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        self.list_sort_var = tk.StringVar(value="emission")
        ttk.Radiobutton(ctrl, text=self.i18n.get_text("sort_by_emission", "Sort by Emission"),
                        variable=self.list_sort_var, value="emission").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ctrl, text=self.i18n.get_text("sort_by_price", "Sort by Price"),
                        variable=self.list_sort_var, value="price").pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl, text=self.i18n.get_text("limit", "Limit") + ":").pack(side=tk.LEFT, padx=5)
        self.list_limit_var = tk.IntVar(value=10)
        tk.Spinbox(ctrl, from_=1, to=50, textvariable=self.list_limit_var, width=4).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl, text=self.i18n.get_text("refresh", "Refresh"),
                   command=self._refresh_subnet_list).pack(side=tk.LEFT, padx=10)

        # Search
        tk.Label(ctrl, text=self.i18n.get_text("search", "Search") + ":").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        tk.Entry(ctrl, textvariable=self.search_var, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text=self.i18n.get_text("search", "Search"),
                   command=self._refresh_subnet_list).pack(side=tk.LEFT, padx=5)

        # Treeview
        cols = ("netuid", "name", "price", "emission")
        self.subnet_tree = ttk.Treeview(tab, columns=cols, show="headings", height=18)
        for c, w in zip(cols, (80, 200, 150, 150)):
            self.subnet_tree.heading(c, text=self.i18n.get_text(c, c.title()))
            self.subnet_tree.column(c, width=w, anchor=tk.CENTER)
        self.subnet_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=self.subnet_tree.yview)
        self.subnet_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # -------------------- Tab 2: Subnet Info --------------------------

    def _create_subnet_info_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.i18n.get_text("tab_subnet_info", "Subnet Info"))

        ctrl = ttk.LabelFrame(tab, text=self.i18n.get_text("options", "Options"))
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(ctrl, text=self.i18n.get_text("netuid", "Subnet UID") + ":").pack(side=tk.LEFT, padx=5)
        self.info_netuid_var = tk.IntVar(value=1)
        tk.Spinbox(ctrl, from_=1, to=999, textvariable=self.info_netuid_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text=self.i18n.get_text("load", "Load"),
                   command=self._load_subnet_info).pack(side=tk.LEFT, padx=10)

        self.info_text = tk.Text(tab, wrap=tk.WORD, state=tk.DISABLED, height=25)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # -------------------- Tab 3: Price Prediction ---------------------

    def _create_price_prediction_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.i18n.get_text("tab_price_prediction", "Price Prediction"))

        # Control panel
        control_frame = ttk.LabelFrame(tab, text=self.i18n.get_text("options", "Options"))
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Subnet selection
        tk.Label(control_frame, text=self.i18n.get_text("select_subnet", "Select Subnet:")).pack(side=tk.LEFT, padx=5)
        self.prediction_subnet_var = tk.StringVar()
        self.prediction_subnet_selector = ttk.Combobox(control_frame, textvariable=self.prediction_subnet_var, width=10)
        self.prediction_subnet_selector.pack(side=tk.LEFT, padx=5)

        # Prediction days
        tk.Label(control_frame, text=self.i18n.get_text("prediction_days", "Days:")).pack(side=tk.LEFT, padx=5)
        self.prediction_days_var = tk.IntVar(value=30)
        days_spinbox = tk.Spinbox(control_frame, from_=7, to=365, textvariable=self.prediction_days_var, width=5)
        days_spinbox.pack(side=tk.LEFT, padx=5)

        # Model selection
        tk.Label(control_frame, text=self.i18n.get_text("prediction_model", "Model:")).pack(side=tk.LEFT, padx=5)
        self.prediction_model_var = tk.StringVar(value="random_forest")
        models = [
            ("random_forest", "Random Forest"),
            ("linear", "Linear Regression"),
            ("svr", "SVR"),
            ("lstm", "LSTM"),
            ("arima", "ARIMA"),
            ("xgboost", "XGBoost"),
            ("prophet", "Prophet")
        ]
        model_selector = ttk.Combobox(control_frame, textvariable=self.prediction_model_var,
                                      values=[m[0] for m in models], state="readonly", width=15)
        model_selector.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text=self.i18n.get_text("predict", "Predict"),
                   command=self._run_prediction).pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text=self.i18n.get_text("save_chart", "Save Chart"),
                   command=self._save_prediction_chart).pack(side=tk.LEFT, padx=5)

        # Info frame
        info_frame = ttk.LabelFrame(tab, text=self.i18n.get_text("prediction_info", "Prediction Info"))
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.prediction_info_var = tk.StringVar(value="")
        tk.Label(info_frame, textvariable=self.prediction_info_var, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=5)

        # Chart area
        chart_frame = ttk.Frame(tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.pred_figure = plt.Figure(figsize=(9, 4), dpi=100)
        self.pred_ax = self.pred_figure.add_subplot(111)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_figure, master=chart_frame)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Populate subnet selector
        self._populate_prediction_subnets()

    # -------------------- Tab 4: Subnet Comparison --------------------

    def _create_subnet_comparison_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.i18n.get_text("tab_subnet_comparison", "Subnet Comparison"))

        ctrl = ttk.LabelFrame(tab, text=self.i18n.get_text("options", "Options"))
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(ctrl, text=self.i18n.get_text("add_subnet", "Add Subnet") + ":").pack(side=tk.LEFT, padx=5)
        self.compare_netuid_var = tk.IntVar(value=1)
        tk.Spinbox(ctrl, from_=1, to=999, textvariable=self.compare_netuid_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text=self.i18n.get_text("add_subnet", "Add"),
                   command=self._add_compare_subnet).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text=self.i18n.get_text("remove_subnet", "Remove"),
                   command=self._remove_compare_subnet).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl, text=self.i18n.get_text("prediction_days", "Days:")).pack(side=tk.LEFT, padx=5)
        self.compare_days_var = tk.IntVar(value=30)
        tk.Spinbox(ctrl, from_=7, to=365, textvariable=self.compare_days_var, width=5).pack(side=tk.LEFT, padx=5)

        self.compare_history_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text=self.i18n.get_text("show_history", "Show History"),
                        variable=self.compare_history_var).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl, text=self.i18n.get_text("compare", "Compare"),
                   command=self._run_comparison).pack(side=tk.LEFT, padx=10)

        # Listbox of selected subnets
        list_frame = ttk.Frame(tab)
        list_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(list_frame, text="Selected subnets:").pack(side=tk.LEFT, padx=5)
        self.compare_listbox = tk.Listbox(list_frame, height=3, width=40)
        self.compare_listbox.pack(side=tk.LEFT, padx=5)
        self.compare_netuids: List[int] = []

        # Chart
        chart_frame = ttk.Frame(tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.cmp_figure = plt.Figure(figsize=(9, 4), dpi=100)
        self.cmp_ax = self.cmp_figure.add_subplot(111)
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_figure, master=chart_frame)
        self.cmp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------- Tab 5: Investment Advice --------------------

    def _create_investment_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.i18n.get_text("tab_investment_recommendation", "Investment Advice"))

        ctrl = ttk.LabelFrame(tab, text=self.i18n.get_text("options", "Options"))
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(ctrl, text=self.i18n.get_text("limit", "Limit") + ":").pack(side=tk.LEFT, padx=5)
        self.rec_limit_var = tk.IntVar(value=5)
        tk.Spinbox(ctrl, from_=1, to=20, textvariable=self.rec_limit_var, width=4).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text=self.i18n.get_text("generate_recommendations", "Generate Recommendations"),
                   command=self._generate_recommendations).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text=self.i18n.get_text("export_csv", "Export CSV"),
                   command=self._export_recommendations_csv).pack(side=tk.LEFT, padx=5)

        cols = ("netuid", "name", "investment_score", "price", "price_change", "recommendation_reason")
        self.rec_tree = ttk.Treeview(tab, columns=cols, show="headings", height=12)
        headers = {
            "netuid": (self.i18n.get_text("netuid", "UID"), 60),
            "name": (self.i18n.get_text("name", "Name"), 140),
            "investment_score": (self.i18n.get_text("investment_score", "Score"), 90),
            "price": (self.i18n.get_text("price", "Price"), 120),
            "price_change": (self.i18n.get_text("price_change", "Change %"), 100),
            "recommendation_reason": (self.i18n.get_text("recommendation_reason", "Reason"), 400),
        }
        for c, (hdr, w) in headers.items():
            self.rec_tree.heading(c, text=hdr)
            self.rec_tree.column(c, width=w, anchor=tk.CENTER if c != "recommendation_reason" else tk.W)
        self.rec_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # -------------------- Tab 6: Miner Recommendations ----------------

    def _create_mining_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.i18n.get_text("tab_miner_recommendation", "Miner Recommendations"))

        ctrl = ttk.LabelFrame(tab, text=self.i18n.get_text("options", "Options"))
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(ctrl, text=self.i18n.get_text("limit", "Limit") + ":").pack(side=tk.LEFT, padx=5)
        self.mine_limit_var = tk.IntVar(value=5)
        tk.Spinbox(ctrl, from_=1, to=20, textvariable=self.mine_limit_var, width=4).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl, text=self.i18n.get_text("gpu_clusters", "GPU Clusters") + ":").pack(side=tk.LEFT, padx=5)
        self.mine_gpu_clusters_var = tk.DoubleVar(value=1.0)
        tk.Spinbox(ctrl, from_=0.1, to=100.0, increment=0.1, textvariable=self.mine_gpu_clusters_var, width=6).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl, text=self.i18n.get_text("daily_cost_tao", "Daily Cost (TAO)") + ":").pack(side=tk.LEFT, padx=5)
        self.mine_daily_cost_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(ctrl, from_=0.0, to=10000.0, increment=0.1, textvariable=self.mine_daily_cost_var, width=7).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl, text=self.i18n.get_text("generate_recommendations", "Generate Recommendations"),
                   command=self._generate_mining_recommendations).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text=self.i18n.get_text("export_csv", "Export CSV"),
                   command=self._export_mining_recommendations_csv).pack(side=tk.LEFT, padx=5)

        cols = ("netuid", "name", "mining_score", "miner_share_pct", "price", "price_change", "recommendation_reason")
        self.mine_tree = ttk.Treeview(tab, columns=cols, show="headings", height=12)
        headers = {
            "netuid": (self.i18n.get_text("netuid", "UID"), 60),
            "name": (self.i18n.get_text("name", "Name"), 140),
            "mining_score": (self.i18n.get_text("mining_score", "Mining Score"), 120),
            "miner_share_pct": (self.i18n.get_text("miner_share", "Share %"), 90),
            "price": (self.i18n.get_text("price", "Price"), 120),
            "price_change": (self.i18n.get_text("price_change", "Change %"), 100),
            "recommendation_reason": (self.i18n.get_text("recommendation_reason", "Reason"), 320),
        }
        for c, (hdr, w) in headers.items():
            self.mine_tree.heading(c, text=hdr)
            self.mine_tree.column(c, width=w, anchor=tk.CENTER if c != "recommendation_reason" else tk.W)
        self.mine_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # ======================= Actions ==================================

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _on_locale_changed(self, _event=None):
        locale = self.locale_var.get()
        self.i18n.set_locale(locale)
        self._set_status(self.i18n.get_text("status_language_changed", "Language changed"))
        # Rebuild UI to reflect new language
        for child in self.root.winfo_children():
            child.destroy()
        self._build_ui()
        self._set_status(self.i18n.get_text("status_ready", "Ready"))

    # --- Subnet list ---

    def _refresh_subnet_list(self):
        self._set_status(self.i18n.get_text("status_loading", "Loading data..."))
        self.root.update_idletasks()

        try:
            sort_by = self.list_sort_var.get()
            limit = self.list_limit_var.get()
            if sort_by == "emission":
                subnets = self.comparison_analyzer.get_top_subnets_by_emission(limit=limit)
            else:
                subnets = self.comparison_analyzer.get_top_subnets_by_price(limit=limit)

            # Apply search filter
            query = self.search_var.get().strip().lower()
            if query:
                subnets = [s for s in subnets
                           if query in str(s.get("netuid", "")).lower()
                           or query in s.get("name", "").lower()]

            self.subnets = subnets

            # Update treeview
            for item in self.subnet_tree.get_children():
                self.subnet_tree.delete(item)
            for s in subnets:
                price = self.data_fetcher.get_subnet_dtao_price(s["netuid"])
                self.subnet_tree.insert("", tk.END, values=(
                    s.get("netuid", ""),
                    s.get("name", ""),
                    f"{price:.6f}" if price else "N/A",
                    f"{s.get('emission', 0):.6f}",
                ))

            self._set_status(self.i18n.get_text("status_ready", "Ready"))
        except Exception as e:
            self._set_status(f"{self.i18n.get_text('status_error', 'Error')}: {e}")

    # --- Subnet info ---

    def _load_subnet_info(self):
        self._set_status(self.i18n.get_text("status_loading", "Loading data..."))
        self.root.update_idletasks()

        try:
            netuid = self.info_netuid_var.get()
            metrics = self.data_fetcher.get_subnet_metrics(netuid)
            price = self.data_fetcher.get_subnet_dtao_price(netuid)

            lines = [
                f"{self.i18n.get_text('basic_info', 'Basic Information')}",
                f"  {self.i18n.get_text('netuid', 'UID')}: {netuid}",
                f"  {self.i18n.get_text('price', 'Price')}: {price:.6f} TAO" if price else "  Price: N/A",
                f"  {self.i18n.get_text('emission', 'Emission')}: {metrics.get('emission', 0):.6f}",
                "",
                f"{self.i18n.get_text('liquidity_info', 'Liquidity Info')}",
                f"  {self.i18n.get_text('tau_in', 'TAO Reserve')}: {metrics.get('tau_in', 0):.6f}",
                f"  {self.i18n.get_text('alpha_in', 'Alpha Reserve')}: {metrics.get('alpha_in', 0):.6f}",
                f"  {self.i18n.get_text('alpha_out', 'Alpha Circulation')}: {metrics.get('alpha_out', 0):.6f}",
                "",
                f"{self.i18n.get_text('subnet_metrics', 'Subnet Metrics')}",
                f"  {self.i18n.get_text('validators', 'Validators')}: {metrics.get('active_validators', 0)}",
                f"  {self.i18n.get_text('miners', 'Miners')}: {metrics.get('active_miners', 0)}",
                f"  Total Stake: {metrics.get('total_stake', 0):.6f}",
                f"  Tempo: {metrics.get('tempo', 0)}",
            ]

            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "\n".join(lines))
            self.info_text.config(state=tk.DISABLED)

            self._set_status(self.i18n.get_text("status_ready", "Ready"))
        except Exception as e:
            self._set_status(f"{self.i18n.get_text('status_error', 'Error')}: {e}")

    # --- Price prediction ---

    def _populate_prediction_subnets(self):
        subnets = self.data_fetcher.get_subnets_list()
        values = [str(s["netuid"]) for s in subnets]
        self.prediction_subnet_selector["values"] = values
        if values:
            self.prediction_subnet_var.set(values[0])

    def _run_prediction(self):
        self._set_status(self.i18n.get_text("status_loading", "Loading data..."))
        self.root.update_idletasks()

        try:
            netuid = int(self.prediction_subnet_var.get())
            days = self.prediction_days_var.get()
            model = self.prediction_model_var.get()

            result = self.price_predictor.predict_future_prices(netuid, days=days, model_name=model)

            if not result.get("success", False):
                messagebox.showerror(self.i18n.get_text("error", "Error"),
                                     result.get("error", "Prediction failed"))
                return

            # Update info label
            current = result["current_price"]
            pred_df = result["prediction"]
            last_pred = pred_df["predicted_price"].iloc[-1]
            change = result["price_change_percent"]
            metrics = result.get("metrics", {})

            info = (
                f"{self.i18n.get_text('current_price', 'Current Price')}: {current:.6f} TAO\n"
                f"{self.i18n.get_text('predicted_price', 'Predicted Price')}: {last_pred:.6f} TAO\n"
                f"{self.i18n.get_text('price_change', 'Price Change')}: {change:.2f}%\n"
                f"{self.i18n.get_text('prediction_model', 'Model')}: {result.get('model_name', '')}\n"
                f"R²: {metrics.get('train_r2', 0):.4f}  MSE: {metrics.get('train_mse', 0):.6f}"
            )
            self.prediction_info_var.set(info)

            # Draw chart
            self.pred_ax.clear()
            full = result["full_data"]
            if "historical_price" in full.columns:
                hist = full["historical_price"].dropna()
                self.pred_ax.plot(hist.index, hist.values, label="Historical", color="blue")
            pred_data = full[full["predicted_price"].notna()]
            self.pred_ax.plot(pred_data.index, pred_data["predicted_price"], label="Predicted", color="red")
            self.pred_ax.fill_between(pred_data.index, pred_data["lower_bound"], pred_data["upper_bound"],
                                      color="red", alpha=0.15)
            self.pred_ax.set_title(f"Subnet {netuid} dTAO Price Prediction")
            self.pred_ax.set_xlabel("Date")
            self.pred_ax.set_ylabel("Price (TAO)")
            self.pred_ax.legend()
            self.pred_ax.grid(True, linestyle="--", alpha=0.7)
            self.pred_figure.autofmt_xdate()
            self.pred_canvas.draw()

            self._last_prediction = result
            self._set_status(self.i18n.get_text("status_ready", "Ready"))
        except Exception as e:
            self._set_status(f"{self.i18n.get_text('status_error', 'Error')}: {e}")

    def _save_prediction_chart(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.pred_figure.savefig(path)
            self._set_status(f"Chart saved to {path}")

    # --- Comparison ---

    def _add_compare_subnet(self):
        netuid = self.compare_netuid_var.get()
        if netuid not in self.compare_netuids:
            self.compare_netuids.append(netuid)
            self.compare_listbox.insert(tk.END, f"Subnet {netuid}")

    def _remove_compare_subnet(self):
        sel = self.compare_listbox.curselection()
        if sel:
            idx = sel[0]
            self.compare_listbox.delete(idx)
            self.compare_netuids.pop(idx)

    def _run_comparison(self):
        if len(self.compare_netuids) < 2:
            messagebox.showwarning("Warning", "Please add at least 2 subnets to compare.")
            return

        self._set_status(self.i18n.get_text("status_loading", "Loading data..."))
        self.root.update_idletasks()

        try:
            days = self.compare_days_var.get()
            result = self.comparison_analyzer.compare_price_predictions(self.compare_netuids, days=days)

            if not result.get("success", False):
                messagebox.showerror(self.i18n.get_text("error", "Error"),
                                     result.get("error", "Comparison failed"))
                return

            self.cmp_ax.clear()
            predictions = result.get("predictions", {})
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(predictions), 1)))

            for i, (netuid, pred) in enumerate(predictions.items()):
                full = pred["full_data"]
                if self.compare_history_var.get() and "historical_price" in full.columns:
                    hist = full["historical_price"].dropna()
                    self.cmp_ax.plot(hist.index, hist.values, "--", color=colors[i], alpha=0.5,
                                     label=f"Subnet {netuid} History")
                pred_data = full[full["predicted_price"].notna()]
                self.cmp_ax.plot(pred_data.index, pred_data["predicted_price"], "-", color=colors[i],
                                 linewidth=2, label=f"Subnet {netuid} Prediction")

            self.cmp_ax.set_title(f"{days} Day Subnet dTAO Price Prediction Comparison")
            self.cmp_ax.set_xlabel("Date")
            self.cmp_ax.set_ylabel("Price (TAO)")
            self.cmp_ax.legend()
            self.cmp_ax.grid(True, linestyle="--", alpha=0.7)
            self.cmp_figure.autofmt_xdate()
            self.cmp_canvas.draw()

            self._set_status(self.i18n.get_text("status_ready", "Ready"))
        except Exception as e:
            self._set_status(f"{self.i18n.get_text('status_error', 'Error')}: {e}")

    # --- Investment recommendations ---

    def _generate_recommendations(self):
        self._set_status(self.i18n.get_text("status_loading", "Loading data..."))
        self.root.update_idletasks()

        try:
            limit = self.rec_limit_var.get()
            recs = self.comparison_analyzer.generate_investment_recommendations(limit=limit)
            self._last_recommendations = recs

            for item in self.rec_tree.get_children():
                self.rec_tree.delete(item)

            for rec in recs:
                self.rec_tree.insert("", tk.END, values=(
                    rec.get("netuid", ""),
                    rec.get("name", ""),
                    f"{rec.get('investment_score', 0):.2f}",
                    f"{rec.get('price', 0):.6f}",
                    f"{rec.get('price_change_percent', 0):.2f}%",
                    rec.get("recommendation_reason", ""),
                ))

            self._set_status(self.i18n.get_text("status_ready", "Ready"))
        except Exception as e:
            self._set_status(f"{self.i18n.get_text('status_error', 'Error')}: {e}")

    def _export_recommendations_csv(self):
        if not self._last_recommendations:
            messagebox.showwarning(
                self.i18n.get_text("warning", "Warning"),
                self.i18n.get_text("no_recommendations_to_export", "Generate recommendations first.")
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return

        fieldnames = [
            "netuid",
            "name",
            "investment_score",
            "price",
            "emission",
            "price_change_percent",
            "active_validators",
            "active_miners",
            "recommendation_reason",
        ]
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._last_recommendations:
                writer.writerow(row)

        self._set_status(f"Recommendations exported to {path}")

    # --- Mining recommendations ---

    def _generate_mining_recommendations(self):
        self._set_status(self.i18n.get_text("status_loading", "Loading data..."))
        self.root.update_idletasks()

        try:
            limit = self.mine_limit_var.get()
            gpu_clusters = self.mine_gpu_clusters_var.get()
            daily_cost_tao = self.mine_daily_cost_var.get()
            recs = self.comparison_analyzer.generate_mining_recommendations(
                limit=limit,
                gpu_clusters=gpu_clusters,
                daily_cluster_cost_tao=daily_cost_tao,
            )
            self._last_mining_recommendations = recs

            for item in self.mine_tree.get_children():
                self.mine_tree.delete(item)

            for rec in recs:
                self.mine_tree.insert("", tk.END, values=(
                    rec.get("netuid", ""),
                    rec.get("name", ""),
                    f"{rec.get('mining_profitability_score', 0):.2f}",
                    f"{rec.get('expected_miner_share_pct', 0):.2f}%",
                    f"{rec.get('price', 0):.6f}",
                    f"{rec.get('price_change_percent', 0):.2f}%",
                    rec.get("recommendation_reason", ""),
                ))

            self._set_status(self.i18n.get_text("status_ready", "Ready"))
        except Exception as e:
            self._set_status(f"{self.i18n.get_text('status_error', 'Error')}: {e}")

    def _export_mining_recommendations_csv(self):
        if not self._last_mining_recommendations:
            messagebox.showwarning(
                self.i18n.get_text("warning", "Warning"),
                self.i18n.get_text("no_recommendations_to_export", "Generate recommendations first.")
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return

        fieldnames = [
            "netuid",
            "name",
            "mining_profitability_score",
            "gross_revenue_index",
            "expected_miner_share_pct",
            "emission_per_active_miner",
            "price",
            "emission",
            "price_change_percent",
            "active_validators",
            "active_miners",
            "recommendation_reason",
        ]
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._last_mining_recommendations:
                writer.writerow(row)

        self._set_status(f"Mining recommendations exported to {path}")
