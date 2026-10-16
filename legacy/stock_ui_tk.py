import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import yfinance as yf
from datetime import datetime, timedelta
import threading
import sys
import os
import traceback
import json
import csv
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class StockPredictionUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("📈 Smart Stock Prediction System Pro v2.0")
        self.root.geometry("1800x1100")
        self.root.state('zoomed')
        
        # Live update variables
        self.live_update_running = False
        self.live_update_thread = None
        self.current_ticker = None
        
        # Fonts
        try:
            self.default_font = font.nametofont("TkDefaultFont")
            self.default_font.configure(size=11)
        except:
            self.default_font = ('Arial', 10)
        self.bold_font = ('Arial', 12, 'bold')
        self.title_font = ('Arial', 18, 'bold')
        self.header_font = ('Arial', 14, 'bold')
        
        # Theme settings
        self.current_theme = 'light'
        self.themes = {
            'light': {
                'bg': '#f8f9fa', 'fg': '#212529', 'button_bg': '#007bff',
                'button_fg': 'white', 'card_bg': 'white', 'highlight': '#e9ecef'
            },
            'dark': {
                'bg': '#343a40', 'fg': '#f8f9fa', 'button_bg': '#495057',
                'button_fg': 'white', 'card_bg': '#495057', 'highlight': '#6c757d'
            }
        }
        
        # Initialize variables
        self.data = None
        self.is_processing = False
        self.prediction_history = self.load_history()
        self.news_list = []
        self.last_prediction = None
        
        # UI Variables
        self.live_price_var = tk.StringVar(value="₹0.00")
        self.status_var = tk.StringVar(value="Ready")
        self.stats_labels = {}
        self.ticker_var = None
        self.period_var = None
        self.model_var = None
        self.use_indicators_var = None
        self.auto_update_var = None
        self.prediction_label = None
        self.confidence_var = None
        self.accuracy_var = None
        self.perf_tree = None
        self.news_text = None
        self.history_tree = None
        self.notebook = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.progress = None
        self.progress_frame = None
        self.status_bar = None
        
        # Initialize UI components
        self.setup_styles()
        self.setup_menu()
        self.setup_ui()
        self.apply_theme(self.current_theme)
        
    def safe_ui_update(self, callback, *args, **kwargs):
        """Safely update UI from any thread"""
        def wrapped_callback():
            try:
                if hasattr(self, 'root') and self.root and self.root.winfo_exists():
                    callback(*args, **kwargs)
            except Exception as e:
                print(f"UI update error: {e}")
        
        try:
            self.root.after(0, wrapped_callback)
        except Exception:
            pass
    
    def setup_styles(self):
        """Configure ttk styles"""
        try:
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('Title.TLabel', font=self.title_font)
            style.configure('Header.TLabel', font=self.header_font)
            style.configure('Primary.TButton', font=self.bold_font)
        except Exception as e:
            print(f"Style setup error: {e}")
    
    def apply_theme(self, theme_name: str):
        """Apply theme colors"""
        if theme_name not in self.themes:
            return
        theme = self.themes[theme_name]
        try:
            style = ttk.Style()
            style.configure('TFrame', background=theme['bg'])
            style.configure('TLabel', background=theme['bg'], foreground=theme['fg'])
            style.configure('TButton', background=theme['button_bg'], foreground=theme['button_fg'])
            self.root.configure(bg=theme['bg'])
        except Exception as e:
            print(f"Theme apply error: {e}")
    
    def setup_menu(self):
        """Setup menu bar"""
        try:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Export Data", command=self.export_data)
            file_menu.add_command(label="Export Predictions", command=self.export_predictions)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.on_closing)
            
            view_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="View", menu=view_menu)
            view_menu.add_command(label="Toggle Dark Mode", command=self.toggle_theme)
            view_menu.add_command(label="Clear History", command=self.clear_history)
            
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="About", command=self.show_about)
        except Exception as e:
            print(f"Menu setup error: {e}")
    
    def toggle_theme(self):
        """Switch between light and dark theme"""
        self.current_theme = 'dark' if self.current_theme == 'light' else 'light'
        self.apply_theme(self.current_theme)
    
    def setup_ui(self):
        """Main UI layout"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Header
        header = ttk.Label(main_frame, text="📈 Smart Stock Prediction System Pro v2.0", 
                          style='Title.TLabel')
        header.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Enhanced Toolbar
        self._setup_enhanced_toolbar(main_frame)
        
        # Sidebar and content
        sidebar = ttk.LabelFrame(main_frame, text="⚙️ Configuration", padding="15")
        sidebar.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(2, weight=1)
        
        self.setup_sidebar(sidebar)
        self.setup_notebook(content_frame)
        
        # Status bar
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress_frame = ttk.Frame(bottom_frame)
        self.progress = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
    
    def _setup_enhanced_toolbar(self, parent):
        """Enhanced toolbar with auto-update controls"""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        ttk.Button(toolbar, text="📥 Fetch Data", command=self.fetch_data_thread).pack(side=tk.LEFT, padx=10)
        ttk.Button(toolbar, text="🎯 Predict", command=self.predict_thread).pack(side=tk.LEFT, padx=10)
        ttk.Button(toolbar, text="📰 News", command=self.fetch_news_thread).pack(side=tk.LEFT, padx=10)
        
        # Auto-update controls
        auto_frame = ttk.Frame(toolbar)
        auto_frame.pack(side=tk.LEFT, padx=10)
        self.auto_update_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(auto_frame, text="🔄 Auto Update", variable=self.auto_update_var,
                       command=self.toggle_auto_update).pack(side=tk.LEFT)
        ttk.Label(auto_frame, text="Every 30s").pack(side=tk.LEFT, padx=(5,0))
        
        ttk.Button(toolbar, text="💹 Update Now", command=self.update_live_price).pack(side=tk.LEFT, padx=10)
        
        # Live price display
        live_frame = ttk.Frame(toolbar)
        live_frame.pack(side=tk.RIGHT)
        ttk.Label(live_frame, text="Live Price:", font=self.bold_font).pack(side=tk.LEFT)
        ttk.Label(live_frame, textvariable=self.live_price_var, font=self.bold_font).pack(side=tk.LEFT, padx=5)
    
    def toggle_auto_update(self):
        """Toggle automatic live updates"""
        if self.auto_update_var.get():
            self.start_auto_update()
        else:
            self.stop_auto_update()
    
    def start_auto_update(self):
        """Start automatic price updates"""
        if self.live_update_running or not self.ticker_var:
            return
            
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            self.auto_update_var.set(False)
            messagebox.showwarning("Error", "Please enter and validate a ticker symbol first")
            return
        
        self.current_ticker = ticker
        self.live_update_running = True
        self.live_update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
        self.live_update_thread.start()
        self.update_status("🔄 Auto-update started")
    
    def stop_auto_update(self):
        """Stop automatic updates"""
        self.live_update_running = False
        self.update_status("⏹️ Auto-update stopped")
    
    def _auto_update_loop(self):
        """Background loop for live updates"""
        while self.live_update_running:
            try:
                self.update_live_price_silent()
                threading.Event().wait(30)
            except Exception as e:
                print(f"Auto-update error: {e}")
                break
    
    def update_live_price_silent(self):
        """Silent live price update"""
        ticker = self.current_ticker
        if not ticker:
            return
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
            
            if price is not None and price > 0:
                price_str = f"₹{float(price):.2f}"
                self.safe_ui_update(lambda p=price_str: self.live_price_var.set(p))
        except Exception as e:
            print(f"Silent update error: {e}")
    
    def setup_sidebar(self, parent):
        """Configuration sidebar"""
        row = 0
        ttk.Label(parent, text="Stock Symbol", font=self.header_font).grid(row=row, sticky=tk.W, pady=10)
        row += 1
        
        self.ticker_var = tk.StringVar(value="RELIANCE.NS")
        ticker_entry = ttk.Entry(parent, textvariable=self.ticker_var, font=self.bold_font, width=20)
        ticker_entry.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Button(parent, text="✅ Validate", command=self.validate_ticker).grid(row=row, pady=10)
        row += 1
        
        ttk.Label(parent, text="Period", font=self.header_font).grid(row=row, sticky=tk.W, pady=10)
        row += 1
        
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(parent, textvariable=self.period_var,
                                   values=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                                   state="readonly")
        period_combo.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Label(parent, text="Model", font=self.header_font).grid(row=row, sticky=tk.W, pady=10)
        row += 1
        
        self.model_var = tk.StringVar(value="random_forest")
        model_combo = ttk.Combobox(parent, textvariable=self.model_var,
                                  values=["random_forest"],
                                  state="readonly")
        model_combo.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        self.use_indicators_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="📊 Technical Indicators", 
                       variable=self.use_indicators_var).grid(row=row, sticky=tk.W)
        row += 1
        
        parent.columnconfigure(0, weight=1)
        
        self.update_history_display()
    
    def setup_notebook(self, parent):
        """Setup tabbed interface"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        self.create_charts_tab()
        self.create_prediction_tab()
        self.create_news_tab()
        self.create_history_tab()
    
    def create_charts_tab(self):
        """Charts tab - FIXED Key Metrics"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Charts & Analysis")
        
        chart_container = ttk.Frame(tab)
        chart_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Stats panel - FIXED: Proper initialization
        stats_frame = ttk.LabelFrame(tab, text="📈 Key Metrics", padding=15)
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        metrics = [
            ("Current Price", "₹0.00"),
            ("Daily Change", "0.00%"),
            ("Period Return", "0.00%"),
            ("Volume", "0"),
            ("Volatility", "0.00%")
        ]
        
        for label_text, default in metrics:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=f"{label_text}:", font=self.bold_font, width=15).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            label_widget = ttk.Label(frame, textvariable=var, font=self.bold_font)
            label_widget.pack(side=tk.RIGHT)
            self.stats_labels[label_text] = var
    
    def create_prediction_tab(self):
        """Prediction tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎯 Predictions")
        
        pred_card = ttk.LabelFrame(tab, text="AI Trading Signal", padding=20)
        pred_card.pack(fill=tk.X, padx=10, pady=10)
        
        self.prediction_label = ttk.Label(pred_card, 
                                         text="Click 'Predict' to generate AI signal", 
                                         font=self.title_font)
        self.prediction_label.pack(pady=20)
        
        details_frame = ttk.Frame(pred_card)
        details_frame.pack(fill=tk.X, pady=10)
        
        self.confidence_var = tk.StringVar(value="Confidence: N/A")
        ttk.Label(details_frame, textvariable=self.confidence_var, font=self.bold_font).pack(side=tk.LEFT)
        
        self.accuracy_var = tk.StringVar(value="Accuracy: N/A")
        ttk.Label(details_frame, textvariable=self.accuracy_var, font=self.bold_font).pack(side=tk.LEFT, padx=20)
        
        perf_frame = ttk.LabelFrame(tab, text="Model Performance", padding=10)
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('Metric', 'Value')
        self.perf_tree = ttk.Treeview(perf_frame, columns=columns, show='headings', height=6)
        for col in columns:
            self.perf_tree.heading(col, text=col)
        self.perf_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_news_tab(self):
        """News tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📰 News")
        
        self.news_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=('Arial', 10))
        self.news_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_history_tab(self):
        """History tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📜 History")
        
        columns = ('Date', 'Ticker', 'Action', 'Confidence', 'Outcome')
        self.history_tree = ttk.Treeview(tab, columns=columns, show='headings', height=15)
        for col in columns:
            self.history_tree.heading(col, text=col)
        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def update_key_metrics(self, data):
        """FIXED: Update key metrics with robust error handling"""
        try:
            if not self.stats_labels or data is None or data.empty:
                print("No stats_labels or empty data")
                return
            
            # Ensure data has required columns
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                print("Missing required columns")
                return
            
            # Get last row safely
            try:
                last_row = data.iloc[-1]
                current_price = pd.to_numeric(last_row['Close'], errors='coerce')
                if pd.isna(current_price) or current_price <= 0:
                    current_price = 0.0
                else:
                    current_price = float(current_price)
            except:
                current_price = 0.0
            
            # Update current price
            self.stats_labels["Current Price"].set(f"₹{current_price:.2f}")
            
            # Daily Change - FIXED
            if len(data) > 1:
                try:
                    prev_row = data.iloc[-2]
                    prev_price = pd.to_numeric(prev_row['Close'], errors='coerce')
                    if pd.isna(prev_price) or prev_price <= 0:
                        prev_price = current_price
                    else:
                        prev_price = float(prev_price)
                    
                    if prev_price > 0:
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        self.stats_labels["Daily Change"].set(f"{change_pct:+.2f}%")
                    else:
                        self.stats_labels["Daily Change"].set("0.00%")
                except:
                    self.stats_labels["Daily Change"].set("0.00%")
            else:
                self.stats_labels["Daily Change"].set("0.00%")
            
            # Period Return - FIXED
            try:
                first_row = data.iloc[0]
                first_price = pd.to_numeric(first_row['Close'], errors='coerce')
                if pd.isna(first_price) or first_price <= 0:
                    first_price = current_price
                else:
                    first_price = float(first_price)
                
                if first_price > 0:
                    period_ret = ((current_price / first_price) - 1) * 100
                    self.stats_labels["Period Return"].set(f"{period_ret:+.2f}%")
                else:
                    self.stats_labels["Period Return"].set("0.00%")
            except:
                self.stats_labels["Period Return"].set("0.00%")
            
            # Volume - FIXED
            try:
                volume = pd.to_numeric(last_row['Volume'], errors='coerce')
                if pd.isna(volume):
                    volume = 0
                else:
                    volume = int(float(volume))
                self.stats_labels["Volume"].set(f"{volume:,}")
            except:
                self.stats_labels["Volume"].set("0")
            
            # Volatility - FIXED
            try:
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 1:
                    vol = returns.std() * np.sqrt(252) * 100
                    vol = float(vol) if not pd.isna(vol) else 0.0
                    self.stats_labels["Volatility"].set(f"{vol:.2f}%")
                else:
                    self.stats_labels["Volatility"].set("0.00%")
            except:
                self.stats_labels["Volatility"].set("0.00%")
                
            print(f"Metrics updated: Price={current_price:.2f}")
            
        except Exception as e:
            print(f"Metrics update error: {e}")
            # Set default values on error
            for key in self.stats_labels:
                self.stats_labels[key].set("N/A")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI with proper handling"""
        try:
            if not isinstance(prices, pd.Series):
                return pd.Series([50.0] * len(prices), index=prices.index if hasattr(prices, 'index') else None)
            
            prices = prices.copy()
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return pd.Series([50.0] * len(prices), index=prices.index)
    
    def update_charts_display(self, data):
        """Update charts with error handling"""
        try:
            if not hasattr(self, 'ax') or data.empty or 'Close' not in data.columns:
                print("Cannot update charts: missing data or components")
                return
            
            self.ax.clear()
            
            close_prices = data['Close'].dropna()
            if len(close_prices) > 0:
                self.ax.plot(close_prices.index, close_prices.values, 
                           label='Close Price', linewidth=2, color='#007bff')
                self.ax.set_title(f"{self.ticker_var.get() if self.ticker_var else 'Stock'} - Price Chart", 
                                fontweight='bold')
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
                self.canvas.draw()
            
            # Update metrics
            self.update_key_metrics(data)
            
            if self.notebook:
                self.notebook.select(0)
                
        except Exception as e:
            print(f"Chart update error: {e}")
    
    def validate_ticker(self):
        """Validate ticker symbol"""
        if not self.ticker_var:
            messagebox.showwarning("Error", "Ticker variable not initialized")
            return
            
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showwarning("Error", "Enter a ticker symbol")
            return
        
        self.show_progress(True)
        self.update_status(f"Validating {ticker}...")
        
        def validate():
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if info and (info.get('symbol') or info.get('shortName')):
                    name = info.get('longName') or info.get('shortName', ticker)
                    self.safe_ui_update(lambda n=name: messagebox.showinfo("Success", f"✓ Valid: {n}"))
                else:
                    self.safe_ui_update(lambda: messagebox.showerror("Error", "Invalid ticker symbol"))
            except Exception as e:
                self.safe_ui_update(lambda: messagebox.showerror("Error", f"Validation failed: {str(e)}"))
            finally:
                self.show_progress(False)
                self.update_status("Ready")
        
        threading.Thread(target=validate, daemon=True).start()
    
    def fetch_data_thread(self):
        """Fetch stock data"""
        if self.is_processing or not self.ticker_var or not self.period_var:
            messagebox.showwarning("Error", "Please configure ticker and period first")
            return
        
        ticker = self.ticker_var.get().strip()
        period = self.period_var.get()
        
        self.is_processing = True
        self.show_progress(True)
        self.update_status(f"Fetching {ticker} data for {period}...")
        
        def fetch():
            try:
                data = yf.download(ticker, period=period, progress=False)
                if data.empty:
                    raise ValueError("No data received from Yahoo Finance")
                
                self.data = data
                print(f"Fetched {len(data)} rows of data")
                self.safe_ui_update(lambda: self.update_charts_display(data))
                
            except Exception as e:
                error_msg = f"Data fetch failed: {str(e)}"
                print(error_msg)
                self.safe_ui_update(lambda msg=error_msg: messagebox.showerror("Data Error", msg))
            finally:
                self.is_processing = False
                self.show_progress(False)
                self.update_status("Ready")
        
        threading.Thread(target=fetch, daemon=True).start()
    
    def fetch_news_thread(self):
        """FIXED: Fetch news with proper error handling"""
        if not self.ticker_var:
            messagebox.showwarning("Error", "Please select a ticker first")
            return
            
        ticker = self.ticker_var.get().strip()
        if not ticker:
            return
        
        self.show_progress(True)
        self.update_status(f"Fetching news for {ticker}...")
        
        def fetch_news():
            try:
                # Try multiple methods to get news
                stock = yf.Ticker(ticker)
                
                # Method 1: Direct news attribute
                news_data = getattr(stock, 'news', [])
                
                # Method 2: If news is empty, try getting recommendations or calendar
                if not news_data:
                    try:
                        recommendations = stock.recommendations
                        if not recommendations.empty:
                            news_data = [{'title': f"Analyst Recommendation Update", 
                                        'publisher': 'Yahoo Finance', 
                                        'providerPublishTime': int(datetime.now().timestamp())}]
                    except:
                        pass
                
                # Filter and format news
                formatted_news = []
                if news_data:
                    for i, item in enumerate(news_data[:5]):
                        try:
                            # Handle different news item formats
                            title = (item.get('title') or 
                                   item.get('uuid', f'News Item {i}') or 
                                   f"Market Update {i+1}")
                            
                            publisher = (item.get('publisher') or 
                                       item.get('source', 'Yahoo Finance') or 
                                       'Market News')
                            
                            timestamp = item.get('providerPublishTime')
                            if timestamp:
                                try:
                                    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                                except:
                                    date_str = 'Recent'
                            else:
                                date_str = 'N/A'
                            
                            formatted_news.append({
                                'title': title[:100] + '...' if len(title) > 100 else title,
                                'publisher': publisher,
                                'date': date_str
                            })
                        except Exception as item_e:
                            print(f"Error processing news item {i}: {item_e}")
                            continue
                
                if not formatted_news:
                    # Fallback message
                    formatted_news = [{
                        'title': f"No recent news available for {ticker}",
                        'publisher': 'Yahoo Finance',
                        'date': datetime.now().strftime('%Y-%m-%d')
                    }]
                
                self.safe_ui_update(lambda news=formatted_news: self.update_news_display(news))
                
            except Exception as e:
                error_msg = f"News fetch failed: {str(e)}"
                print(error_msg)
                fallback_news = [{
                    'title': f"News unavailable: {str(e)}",
                    'publisher': 'System',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }]
                self.safe_ui_update(lambda news=fallback_news: self.update_news_display(news))
            finally:
                self.show_progress(False)
                self.update_status("Ready")
        
        threading.Thread(target=fetch_news, daemon=True).start()
    
    def update_news_display(self, news_items):
        """FIXED: Update news display with proper formatting"""
        try:
            if not self.news_text:
                return
                
            self.news_text.delete(1.0, tk.END)
            
            if not news_items:
                self.news_text.insert(tk.END, "No news available\n")
                return
            
            for item in news_items:
                title = item.get('title', 'No title')
                publisher = item.get('publisher', 'Unknown')
                date = item.get('date', 'N/A')
                
                news_entry = f"📰 {title}\n"
                news_entry += f"📅 {date} | 📻 {publisher}\n"
                news_entry += "=" * 80 + "\n\n"
                
                self.news_text.insert(tk.END, news_entry)
            
            # Scroll to top
            self.news_text.see(1.0)
            
            if self.notebook:
                self.notebook.select(2)
                
        except Exception as e:
            print(f"News display error: {e}")
            if self.news_text:
                self.news_text.delete(1.0, tk.END)
                self.news_text.insert(tk.END, f"Display error: {str(e)}\n")
    
    # Rest of the methods remain the same but with minor fixes...
    def predict_thread(self):
        """Generate prediction"""
        if self.data is None or len(self.data) < 30:
            messagebox.showwarning("Error", "Fetch data first (need at least 30 days)")
            return
        
        self.is_processing = True
        self.show_progress(True)
        self.update_status("Running ML prediction...")
        
        def predict():
            try:
                df = self.data.copy()
                
                # Feature engineering with error handling
                df['Returns'] = df['Close'].pct_change().fillna(0)
                df['MA_5'] = df['Close'].rolling(5).mean().fillna(method='bfill')
                df['MA_20'] = df['Close'].rolling(20).mean().fillna(method='bfill')
                df['RSI'] = self.calculate_rsi(df['Close']).fillna(50)
                df['Volatility'] = df['Returns'].rolling(10).std().fillna(0)
                df['Volume_MA'] = df['Volume'].rolling(10).mean().fillna(0)
                
                df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                df = df.dropna()
                
                if len(df) < 20:
                    raise ValueError("Insufficient data after feature engineering")
                
                features = ['Returns', 'MA_5', 'MA_20', 'RSI', 'Volatility', 'Volume_MA']
                X = df[features].fillna(0)
                y = df['Target']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                last_data = X.iloc[-1:].fillna(0)
                prediction = model.predict(last_data)[0]
                confidence = model.predict_proba(last_data).max()
                
                result = {
                    'action': 'BUY' if prediction == 1 else 'SELL',
                    'confidence': confidence,
                    'accuracy': accuracy,
                    'date': datetime.now().isoformat(),
                    'ticker': self.ticker_var.get() if self.ticker_var else 'Unknown'
                }
                
                self.last_prediction = result
                self.save_to_history(result)
                self.safe_ui_update(lambda r=result: self.display_prediction(r))
                
            except Exception as e:
                self.safe_ui_update(lambda: messagebox.showerror("Prediction Error", str(e)))
            finally:
                self.is_processing = False
                self.show_progress(False)
                self.update_status("Ready")
        
        threading.Thread(target=predict, daemon=True).start()
    
    def display_prediction(self, result):
        """Display prediction results"""
        try:
            if (not self.prediction_label or not self.confidence_var or 
                not self.accuracy_var or not self.perf_tree):
                return
                
            action = result['action']
            confidence = result['confidence']
            accuracy = result['accuracy']
            
            color_map = {'BUY': '#28a745', 'SELL': '#dc3545'}
            color = color_map.get(action, 'black')
            
            self.prediction_label.config(text=f"🎯 {action} SIGNAL", foreground=color)
            self.confidence_var.set(f"Confidence: {confidence:.1%}")
            self.accuracy_var.set(f"Accuracy: {accuracy:.1%}")
            
            self.perf_tree.delete(*self.perf_tree.get_children())
            metrics = [
                ('Model', 'Random Forest'),
                ('Accuracy', f"{accuracy:.1%}"),
                ('Confidence', f"{confidence:.1%}"),
                ('Signal', action)
            ]
            for metric, value in metrics:
                self.perf_tree.insert('', 'end', values=(metric, value))
            
            if self.notebook:
                self.notebook.select(1)
        except Exception as e:
            print(f"Display prediction error: {e}")
    
    def update_live_price(self):
        """Manual live price update"""
        if not self.ticker_var:
            return
        ticker = self.ticker_var.get().strip()
        if not ticker:
            return
        
        def update():
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                price = (info.get('currentPrice') or 
                        info.get('regularMarketPrice') or 
                        info.get('previousClose'))
                if price:
                    price_str = f"₹{float(price):.2f}"
                    self.safe_ui_update(lambda p=price_str: self.live_price_var.set(p))
            except Exception as e:
                print(f"Live price error: {e}")
        
        threading.Thread(target=update, daemon=True).start()
    
    def show_progress(self, show=True):
        """Show/hide progress bar"""
        def toggle():
            if self.progress and self.progress_frame:
                if show:
                    self.progress.start(10)
                    self.progress_frame.pack(fill=tk.X, pady=5)
                else:
                    self.progress.stop()
                    self.progress_frame.pack_forget()
        self.safe_ui_update(toggle)
    
    def update_status(self, message):
        """Update status bar"""
        if self.status_var:
            self.safe_ui_update(lambda m=message: self.status_var.set(m))
    
    # History and file operations (same as before but with better error handling)
    def load_history(self):
        try:
            if os.path.exists('predictions.json'):
                with open('predictions.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"History load error: {e}")
        return []
    
    def save_to_history(self, result):
        entry = {
            'date': result.get('date', datetime.now().isoformat()),
            'ticker': result.get('ticker', 'Unknown'),
            'action': result['action'],
            'confidence': result['confidence'],
            'accuracy': result['accuracy'],
            'outcome': 'Pending'
        }
        self.prediction_history.append(entry)
        try:
            with open('predictions.json', 'w') as f:
                json.dump(self.prediction_history, f, indent=2, default=str)
        except Exception as e:
            print(f"History save error: {e}")
        self.update_history_display()
    
    def update_history_display(self):
        def refresh():
            if not self.history_tree:
                return
            self.history_tree.delete(*self.history_tree.get_children())
            recent_history = self.prediction_history[-20:]
            for entry in recent_history:
                self.history_tree.insert('', 'end', values=(
                    entry.get('date', '')[:16],
                    entry.get('ticker', ''),
                    entry.get('action', ''),
                    f"{entry.get('confidence', 0):.1%}",
                    entry.get('outcome', 'Pending')
                ))
        self.safe_ui_update(refresh)
    
    def clear_history(self):
        self.prediction_history = []
        try:
            with open('predictions.json', 'w') as f:
                json.dump([], f)
        except:
            pass
        self.update_history_display()
        messagebox.showinfo("Cleared", "Prediction history cleared")
    
    def export_data(self):
        if self.data is None or self.data.empty:
            messagebox.showwarning("No Data", "No data to export")
            return
        ticker = self.ticker_var.get() if self.ticker_var else "stock"
        filename = f"{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
        try:
            self.data.to_csv(filename)
            messagebox.showinfo("Success", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def export_predictions(self):
        if not self.prediction_history:
            messagebox.showwarning("No Data", "No predictions to export")
            return
        filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.prediction_history, f, indent=2, default=str)
            messagebox.showinfo("Success", f"Predictions exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def show_about(self):
        messagebox.showinfo("About", 
            "Smart Stock Prediction System Pro v2.0\n"
            "Professional Trading Dashboard with ML Predictions\n"
            "Built with Python, yfinance, and scikit-learn")
    
    def on_closing(self):
        """Cleanup on close"""
        self.stop_auto_update()
        if messagebox.askokcancel("Quit", "Exit application?"):
            if self.fig:
                plt.close(self.fig)
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Start main loop"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = StockPredictionUI()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        print(traceback.format_exc())
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()