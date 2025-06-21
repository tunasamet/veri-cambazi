import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib
import os
import pickle
import io
from PIL import Image, ImageTk
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.cluster import KMeans

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns

class VeriCambazi(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Veri Cambazı")
        self.geometry("1200x750")
        self.configure(bg="#f5f6fa")
        self.df = None
        self.filtered_df = None
        self.filters = []
        self.plot_win = None
        self.undo_stack = []
        self.redo_stack = []
        self.create_widgets()
        
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook.Tab", font=("Segoe UI", 12, "bold"), padding=[12, 8])
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=7)
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"), background="#dff9fb")
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=26, fieldbackground="#f5f6fa", background="#ecf0f1")
        style.map("TButton",
                  foreground=[('pressed', '#273c75'), ('active', '#353b48')],
                  background=[('active', '#dff9fb')])
        style.configure("TLabelframe", background="#f5f6fa")
        style.configure("TLabel", background="#f5f6fa", font=("Segoe UI", 11))
        

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.auto_backup()
        self.destroy()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sekmeler
        self.tab_data = tk.Frame(notebook)
        self.tab_preprocess = tk.Frame(notebook)
        self.tab_analysis = tk.Frame(notebook)
        self.tab_viz = tk.Frame(notebook)
        self.tab_ml = tk.Frame(notebook)

        notebook.add(self.tab_data, text="Veri")
        notebook.add(self.tab_preprocess, text="Ön İşleme")
        notebook.add(self.tab_analysis, text="Analiz")
        notebook.add(self.tab_viz, text="Görselleştirme")
        notebook.add(self.tab_ml, text="Makine Öğrenmesi")

        # Veri sekmesi
        frm = tk.Frame(self.tab_data)
        frm.pack(pady=10)
        tk.Button(frm, text="Veriyi Göster (Pencere)", command=self.show_data_popup).grid(row=0, column=7, padx=5)
        tk.Button(frm, text="Veri Yükle", command=self.load_data).grid(row=0, column=0, padx=5)
        tk.Button(frm, text="⬅ Geri Al", command=self.undo).grid(row=0, column=1, padx=5)
        tk.Button(frm, text="İleri Al ➡", command=self.redo).grid(row=0, column=2, padx=5)
        tk.Button(frm, text="Veriyi Kaydet", command=self.save_filtered_data).grid(row=0, column=3, padx=5)
        tk.Button(frm, text="Yedeği Geri Yükle", command=self.restore_backup).grid(row=0, column=4, padx=5)
        tk.Button(frm, text="Veri Önizle", command=self.show_preview).grid(row=0, column=5, padx=5)
        self.lbl_filters = tk.Label(self.tab_data, text="Uygulanan Filtreler: Yok", fg="blue")
        self.lbl_filters.pack(pady=5)

        # --- Ön İşleme sekmesi ---
        frm2 = tk.Frame(self.tab_preprocess)
        frm2.pack(pady=10)
        tk.Button(frm2, text="Veriyi Filtrele", command=self.filter_data).grid(row=0, column=0, padx=5)
        tk.Button(frm2, text="Filtreleri Temizle", command=self.clear_filters).grid(row=0, column=1, padx=5)
        tk.Button(frm2, text="Kolonları Düzenle", command=self.edit_columns).grid(row=0, column=2, padx=5)
        tk.Button(frm2, text="Satır Sil", command=self.delete_selected_rows).grid(row=0, column=3, padx=5)
        tk.Button(frm2, text="Satır Düzenle", command=self.edit_selected_row).grid(row=0, column=4, padx=5)
        tk.Button(frm2, text="Eksik Değer Doldurma", command=self.suggest_fill_missing_popup).grid(row=0, column=5, padx=5)
        tk.Button(frm2, text="Veri Ölçeklendir", command=self.scale_data).grid(row=0, column=6, padx=5)
        tk.Button(frm2, text="Veri Dönüştür", command=self.data_transform).grid(row=0, column=7, padx=5)

        # --- Analiz sekmesi ---
        frm3 = tk.Frame(self.tab_analysis)
        frm3.pack(pady=10)
        tk.Button(frm3, text="Veri İstatistikleri", command=self.show_stats).grid(row=0, column=0, padx=5)
        tk.Button(frm3, text="Aykırı Değer Analizi", command=self.outlier_analysis_popup).grid(row=0, column=1, padx=5)
        tk.Button(frm3, text="Korelasyon Matrisi", command=self.show_correlation_matrix).grid(row=0, column=2, padx=5)

        # --- Görselleştirme sekmesi ---
        frm4 = tk.Frame(self.tab_viz)
        frm4.pack(pady=10)
        tk.Button(frm4, text="Veriyi Görselleştir", command=self.visualize_data).grid(row=0, column=0, padx=5)

        # --- Makine Öğrenmesi sekmesi ---
        frm5 = tk.Frame(self.tab_ml)
        frm5.pack(pady=10)
        tk.Button(frm5, text="Çoklu Doğrusal Regresyon", command=self.multi_linear_regression_popup).pack(pady=5)
        tk.Button(frm5, text="KNN Sınıflandırma", command=self.knn_classification_popup).pack(pady=5)
        tk.Button(frm5, text="K-Means Kümeleme", command=self.kmeans_clustering_popup).pack(pady=5)
        self.ml_result_text = tk.Text(self.tab_ml, height=18, width=120)
        self.ml_result_text.pack(pady=5)

        # --- Ortak tablo (tüm sekmelerde altta görünür) ---
        self.tree = ttk.Treeview(self)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def multi_linear_regression_popup(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        num_cols = self.filtered_df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) < 2:
            messagebox.showwarning("Uyarı", "En az 2 sayısal kolon olmalı.")
            return
        popup = tk.Toplevel(self)
        popup.title("Çoklu Doğrusal Regresyon")
        tk.Label(popup, text="Hedef (Y) Değişkeni:").grid(row=0, column=0, padx=5, pady=5)
        y_var = tk.StringVar(value=num_cols[0])
        y_menu = ttk.Combobox(popup, textvariable=y_var, values=num_cols, state="readonly")
        y_menu.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(popup, text="Giriş (X) Değişkenleri (Ctrl ile seç):").grid(row=1, column=0, padx=5, pady=5)
        x_listbox = tk.Listbox(popup, selectmode=tk.MULTIPLE, exportselection=0, height=8)
        for col in num_cols:
            x_listbox.insert(tk.END, col)
        x_listbox.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(popup, text="Train/Test Oranı (ör. 0.2 = %20 test):").grid(row=2, column=0, padx=5, pady=5)
        test_size_var = tk.StringVar(value="0.2")
        test_entry = tk.Entry(popup, textvariable=test_size_var)
        test_entry.grid(row=2, column=1, padx=5, pady=5)

        def run_regression():
            y_col = y_var.get()
            x_cols = [x_listbox.get(i) for i in x_listbox.curselection()]
            try:
                test_size = float(test_size_var.get())
                assert 0 < test_size < 1
            except Exception:
                messagebox.showerror("Hata", "Test oranı 0-1 arasında olmalı.")
                return
            if y_col in x_cols:
                messagebox.showerror("Hata", "Hedef değişken giriş değişkenlerinden biri olamaz.")
                return
            if not x_cols or not y_col:
                messagebox.showerror("Hata", "Değişken(ler) seçilmedi.")
                return
            df_nn = self.filtered_df.dropna(subset=[y_col] + x_cols)
            if df_nn.empty:
                messagebox.showerror("Hata", "Seçili kolonlarda yeterli veri yok!")
                return
            X = df_nn[x_cols]
            y = df_nn[y_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            summary = f"""Çoklu Doğrusal Regresyon Sonuçları:
--------------------------------------------------------
Bağımlı (Y): {y_col}
Bağımsız (X): {', '.join(x_cols)}
Test Oranı: {test_size}
Eğitim Kümesi Boyutu: {len(X_train)}
Test Kümesi Boyutu: {len(X_test)}

R2 Skoru:    {r2:.4f}
MAE:         {mae:.4f}
RMSE:        {rmse:.4f}

Katsayılar:
"""
            for xi, coef in zip(x_cols, model.coef_):
                summary += f"  {xi}: {coef:.4f}\n"
            summary += f"Y_Ön_Tahmin = {' + '.join([f'{coef:.4f}*{xi}' for xi, coef in zip(x_cols, model.coef_)])} + {model.intercept_:.4f}\n\n"
            summary += "Gerçek vs Tahmin (ilk 10 satır):\n"
            for yi, yp in zip(y_test[:10], y_pred[:10]):
                summary += f"Gerçek: {yi:.4f} | Tahmin: {yp:.4f}\n"
            # Sonuçları sekmede göster
            self.ml_result_text.delete("1.0", tk.END)
            self.ml_result_text.insert(tk.END, summary)
            # Kapanış
            popup.destroy()

        tk.Button(popup, text="Modeli Eğit & Test Et", command=run_regression).grid(row=3, column=0, columnspan=2, pady=12)
    
    def knn_classification_popup(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        num_cols = self.filtered_df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) < 2:
            messagebox.showwarning("Uyarı", "En az 2 sayısal kolon olmalı.")
            return
        popup = tk.Toplevel(self)
        popup.title("KNN Sınıflandırma")
        tk.Label(popup, text="Hedef (Y) Değişkeni:").grid(row=0, column=0, padx=5, pady=5)
        y_var = tk.StringVar(value=self.filtered_df.columns[0])
        y_menu = ttk.Combobox(popup, textvariable=y_var, values=list(self.filtered_df.columns), state="readonly")
        y_menu.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(popup, text="Giriş (X) Değişkenleri (Ctrl ile seç):").grid(row=1, column=0, padx=5, pady=5)
        x_listbox = tk.Listbox(popup, selectmode=tk.MULTIPLE, exportselection=0, height=8)
        for col in num_cols:
            x_listbox.insert(tk.END, col)
        x_listbox.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(popup, text="Test Oranı (ör. 0.2 = %20 test):").grid(row=2, column=0, padx=5, pady=5)
        test_size_var = tk.StringVar(value="0.2")
        test_entry = tk.Entry(popup, textvariable=test_size_var)
        test_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Label(popup, text="K (Komşu Sayısı):").grid(row=3, column=0, padx=5, pady=5)
        k_var = tk.StringVar(value="5")
        k_entry = tk.Entry(popup, textvariable=k_var)
        k_entry.grid(row=3, column=1, padx=5, pady=5)
        def run_knn():
            y_col = y_var.get()
            x_cols = [x_listbox.get(i) for i in x_listbox.curselection()]
            try:
                test_size = float(test_size_var.get())
                k_val = int(k_var.get())
                assert 0 < test_size < 1 and k_val > 0
            except Exception:
                messagebox.showerror("Hata", "Test oranı ve K uygun formatta olmalı.")
                return
            if y_col in x_cols:
                messagebox.showerror("Hata", "Hedef değişken giriş değişkenlerinden biri olamaz.")
                return
            if not x_cols or not y_col:
                messagebox.showerror("Hata", "Değişken(ler) seçilmedi.")
                return
            df_nn = self.filtered_df.dropna(subset=[y_col] + x_cols)
            if df_nn.empty:
                messagebox.showerror("Hata", "Seçili kolonlarda yeterli veri yok!")
                return
            X = df_nn[x_cols]
            y = df_nn[y_col]
            # Kategorik ise label encoding uygula
            if y.dtype == object or y.dtype == "category":
                y = pd.factorize(y)[0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = KNeighborsClassifier(n_neighbors=k_val)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            summary = f"""KNN Sınıflandırma Sonuçları:
--------------------------------------------------------
Bağımlı (Y): {y_col}
Bağımsız (X): {', '.join(x_cols)}
Test Oranı: {test_size}
K (Komşu): {k_val}
Eğitim Kümesi Boyutu: {len(X_train)}
Test Kümesi Boyutu: {len(X_test)}

Doğruluk (Accuracy): {acc:.4f}

Karışıklık Matrisi:
{cm}

Sınıf Bazlı Rapor:
{report}

Gerçek vs Tahmin (ilk 10 satır):
"""
            for yi, yp in zip(y_test[:10], y_pred[:10]):
                summary += f"Gerçek: {yi} | Tahmin: {yp}\n"
            self.ml_result_text.delete("1.0", tk.END)
            self.ml_result_text.insert(tk.END, summary)
            popup.destroy()
        tk.Button(popup, text="Modeli Eğit & Test Et", command=run_knn).grid(row=4, column=0, columnspan=2, pady=12)    
    
    def kmeans_clustering_popup(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        num_cols = self.filtered_df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) < 2:
            messagebox.showwarning("Uyarı", "En az 2 sayısal kolon olmalı.")
            return
        popup = tk.Toplevel(self)
        popup.title("K-Means Kümeleme")
        tk.Label(popup, text="Kümeleme için Kullanılacak Değişkenler (Ctrl ile seç):").grid(row=0, column=0, padx=5, pady=5)
        x_listbox = tk.Listbox(popup, selectmode=tk.MULTIPLE, exportselection=0, height=8)
        for col in num_cols:
            x_listbox.insert(tk.END, col)
        x_listbox.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(popup, text="Küme Sayısı (K):").grid(row=1, column=0, padx=5, pady=5)
        k_var = tk.StringVar(value="3")
        k_entry = tk.Entry(popup, textvariable=k_var)
        k_entry.grid(row=1, column=1, padx=5, pady=5)
        def run_kmeans():
            x_cols = [x_listbox.get(i) for i in x_listbox.curselection()]
            try:
                k_val = int(k_var.get())
                assert k_val > 1
            except Exception:
                messagebox.showerror("Hata", "K en az 2 ve tam sayı olmalı.")
                return
            if not x_cols or (len(x_cols) < 2 and k_val > 2):
                messagebox.showerror("Hata", "En az iki değişken seçilmeli.")
                return
            df_nn = self.filtered_df.dropna(subset=x_cols)
            if df_nn.empty:
                messagebox.showerror("Hata", "Seçili kolonlarda yeterli veri yok!")
                return
            X = df_nn[x_cols]
            model = KMeans(n_clusters=k_val, random_state=42, n_init='auto' if hasattr(KMeans(), 'n_init') else 10)
            labels = model.fit_predict(X)
            summary = f"""K-Means Kümeleme Sonuçları:
--------------------------------------------------------
Kullanılan Değişkenler: {', '.join(x_cols)}
Küme Sayısı (K): {k_val}
Veri Seti Boyutu: {len(X)}

Küme Merkezleri:
"""
            for i, center in enumerate(model.cluster_centers_):
                summary += f"Küme {i}: {', '.join([f'{val:.4f}' for val in center])}\n"
            summary += "\nHer Kümedeki Örnek Sayısı:\n"
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                summary += f"Küme {label}: {count}\n"
            summary += "\nİlk 10 Satırda Küme Ataması:\n"
            for idx, lbl in zip(df_nn.index[:10], labels[:10]):
                summary += f"Satır {idx}: Küme {lbl}\n"
            # Sonuçları sekmede göster
            self.ml_result_text.delete("1.0", tk.END)
            self.ml_result_text.insert(tk.END, summary)
            popup.destroy()
        tk.Button(popup, text="Kümele", command=run_kmeans).grid(row=2, column=0, columnspan=2, pady=12)
    
    # ---  Satır Silme ---
    def delete_selected_rows(self):
        if self.filtered_df is None or len(self.filtered_df) == 0:
            messagebox.showwarning("Uyarı", "Silinecek veri yok.")
            return
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Uyarı", "Lütfen silmek için en az bir satır seçin.")
            return
        selected_indices = []
        for item in selected_items:
            values = self.tree.item(item, "values")
            mask = (self.filtered_df.astype(str) == values).all(axis=1)
            idx_list = self.filtered_df[mask].index.tolist()
            if idx_list:
                selected_indices.append(idx_list[0])
        if not selected_indices:
            messagebox.showwarning("Uyarı", "Satır(lar) bulunamadı.")
            return
        if not messagebox.askyesno("Onay", f"{len(selected_indices)} satır silinsin mi?"):
            return
        self.push_undo()
        self.filtered_df.drop(index=selected_indices, inplace=True)
        self.filtered_df.reset_index(drop=True, inplace=True)
        self.show_data()
        messagebox.showinfo("Başarılı", "Satır(lar) silindi.")

    # --- Satır Düzenleme ---
    def edit_selected_row(self):
        if self.filtered_df is None or len(self.filtered_df) == 0:
            messagebox.showwarning("Uyarı", "Düzenlenecek veri yok.")
            return
        selected_items = self.tree.selection()
        if len(selected_items) != 1:
            messagebox.showwarning("Uyarı", "Lütfen düzenlemek için bir satır seçin.")
            return
        item = selected_items[0]
        values = self.tree.item(item, "values")
        mask = (self.filtered_df.astype(str) == values).all(axis=1)
        idx_list = self.filtered_df[mask].index.tolist()
        if not idx_list:
            messagebox.showwarning("Uyarı", "Satır bulunamadı.")
            return
        df_idx = idx_list[0]
        edit_popup = tk.Toplevel(self)
        edit_popup.title("Satır Düzenle")
        entries = []
        for i, col in enumerate(self.filtered_df.columns):
            tk.Label(edit_popup, text=col).grid(row=i, column=0, padx=5, pady=5)
            entry = tk.Entry(edit_popup)
            entry.insert(0, str(self.filtered_df.iloc[df_idx, i]))
            entry.grid(row=i, column=1, padx=5, pady=5)
            entries.append(entry)
        def save_edit():
            self.push_undo()
            for j, entry in enumerate(entries):
                val = entry.get()
                #col = self.filtered_df.columns[j]
                old_val = self.filtered_df.iloc[df_idx, j]
                try:
                    if pd.isna(old_val):
                        if val == "":
                            new_val = None
                        else:
                            try:
                                new_val = float(val)
                                if old_val is None or isinstance(old_val, int):
                                    new_val = int(float(val))
                            except Exception:
                                new_val = val
                    elif isinstance(old_val, float):
                        new_val = float(val)
                    elif isinstance(old_val, int):
                        new_val = int(float(val))
                    else:
                        new_val = val
                except Exception:
                    new_val = val
                self.filtered_df.iat[df_idx, j] = new_val
            self.show_data()
            edit_popup.destroy()
            messagebox.showinfo("Başarılı", "Satır güncellendi.")
        tk.Button(edit_popup, text="Kaydet", command=save_edit).grid(row=len(self.filtered_df.columns), column=0, columnspan=2, pady=10)

    # --- 2. Aykırı Değer Analizi (z-score ve IQR) ---
    def outlier_analysis_popup(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Uyarı", "Veri yok.")
            return
        num_cols = self.filtered_df.select_dtypes(include='number').columns
        if len(num_cols) == 0:
            messagebox.showinfo("Bilgi", "Sayısal kolon yok.")
            return
        popup = tk.Toplevel(self)
        popup.title("Aykırı Değer Analizi")
        tk.Label(popup, text="Kolon Seçin:").grid(row=0, column=0, padx=5, pady=5)
        col_var = tk.StringVar(value=num_cols[0])
        col_menu = ttk.Combobox(popup, textvariable=col_var, values=list(num_cols), state="readonly")
        col_menu.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(popup, text="Yöntem:").grid(row=1, column=0, padx=5, pady=5)
        method_var = tk.StringVar(value="z-score")
        method_menu = ttk.Combobox(popup, textvariable=method_var, values=["z-score", "IQR"], state="readonly")
        method_menu.grid(row=1, column=1, padx=5, pady=5)
        result_text = tk.Text(popup, height=6, width=42)
        result_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        def analyze():
            col = col_var.get()
            method = method_var.get()
            vals = self.filtered_df[col].dropna()
            if method == "z-score":
                mean = vals.mean()
                std = vals.std(ddof=0)
                z = (vals - mean) / std
                outlier_idx = z.index[np.abs(z) > 3].tolist()
                msg = f"z-score ile |z|>3 olan {len(outlier_idx)} aykırı değer bulundu."
            else:
                Q1 = vals.quantile(0.25)
                Q3 = vals.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5*IQR
                upper = Q3 + 1.5*IQR
                outlier_idx = vals.index[(vals < lower) | (vals > upper)].tolist()
                msg = f"IQR ile {len(outlier_idx)} aykırı değer bulundu."
            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, msg)
            result_text.insert(tk.END, "\nIndexler: " + ", ".join(map(str, outlier_idx[:10])))
            if len(outlier_idx) > 10:
                result_text.insert(tk.END, f" ... (toplam {len(outlier_idx)})")
            btn_mark.config(command=lambda: mark_outliers(outlier_idx, col))
            btn_delete.config(command=lambda: delete_outliers(outlier_idx))
            btn_mark.config(state=tk.NORMAL)
            btn_delete.config(state=tk.NORMAL)
        def mark_outliers(outlier_idx, col):
            marked_col = f"{col}_outlier"
            self.filtered_df[marked_col] = False
            self.filtered_df.loc[outlier_idx, marked_col] = True
            self.show_data()
            messagebox.showinfo("Bilgi", f"{col} için aykırı değerler '{marked_col}' sütununda işaretlendi.")
        def delete_outliers(outlier_idx):
            if not outlier_idx:
                messagebox.showinfo("Bilgi", "Silinecek aykırı değer yok.")
                return
            self.push_undo()
            self.filtered_df.drop(index=outlier_idx, inplace=True)
            self.filtered_df.reset_index(drop=True, inplace=True)
            self.show_data()
            messagebox.showinfo("Başarılı", f"{len(outlier_idx)} aykırı değer silindi.")
        btn_analyze = tk.Button(popup, text="Analiz Et", command=analyze)
        btn_analyze.grid(row=2, column=0, columnspan=2, pady=5)
        btn_mark = tk.Button(popup, text="Aykırıları İşaretle", state=tk.DISABLED)
        btn_mark.grid(row=3, column=0, padx=5, pady=5)
        btn_delete = tk.Button(popup, text="Aykırıları Sil", state=tk.DISABLED)
        btn_delete.grid(row=3, column=1, padx=5, pady=5)

    # --- 3. Kolonlar Arası Korelasyon Matrisi ve Isı Haritası ---
    def show_correlation_matrix(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Uyarı", "Veri yok.")
            return
        num_cols = self.filtered_df.select_dtypes(include='number').columns
        if len(num_cols) < 2:
            messagebox.showinfo("Bilgi", "Korelasyon için en az 2 sayısal kolon gerekir.")
            return
        corr = self.filtered_df[num_cols].corr()
        popup = tk.Toplevel(self)
        popup.title("Korelasyon Matrisi")
        popup.geometry("700x500")
        tree = ttk.Treeview(popup)
        tree.pack(fill=tk.BOTH, expand=True)
        tree["columns"] = list(corr.columns)
        tree["show"] = "headings"
        for col in corr.columns:
            tree.heading(col, text=col)
            tree.column(col, width=90)
        for idx, row in corr.iterrows():
            vals = [f"{v:.2f}" for v in row]
            tree.insert("", tk.END, values=vals)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Korelasyon Isı Haritası")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize((420, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(popup, image=photo)
        label.image = photo
        label.pack(pady=8)

    # Eksik değerlerin otomatik doldurulması için öneri fonksiyonu
    def suggest_fill_missing_popup(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Uyarı", "Veri yok.")
            return
        popup = tk.Toplevel(self)
        popup.title("Eksik Değer Doldurma")
        popup.geometry("650x400")
        cols_with_na = [col for col in self.filtered_df.columns if self.filtered_df[col].isnull().sum() > 0]
        if not cols_with_na:
            tk.Label(popup, text="Eksik değer yok.").pack(pady=10)
            return
        tree = ttk.Treeview(popup)
        tree.pack(fill=tk.BOTH, expand=True)
        tree["columns"] = ["Kolon", "Eksik (%)", "Uygula"]
        tree["show"] = "headings"
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=180)
        btns = {}
        for col in cols_with_na:
            n_missing = self.filtered_df[col].isnull().sum()
            percent = 100 * n_missing / len(self.filtered_df)
            iid = tree.insert("", tk.END, values=(col, f"{percent:.1f}%", "Ortalama ile doldur"))
            btns[iid] = col
        def on_double_click(event):
            item = tree.selection()
            if not item:
                return
            iid = item[0]
            col = btns[iid]
            self.push_undo()
            # Tüm string, boşluk, "?" vs. NaN olur
            self.filtered_df[col] = pd.to_numeric(self.filtered_df[col], errors="coerce")
            # Ortalama ile doldur
            fill_val = self.filtered_df[col].mean()
            self.filtered_df[col].fillna(fill_val, inplace=True)
            self.show_data()
            messagebox.showinfo("Başarılı", f"{col} kolonundaki eksik ve uygunsuz değerler ortalama ({fill_val}) ile dolduruldu.")
            popup.destroy()
        tree.bind("<Double-1>", on_double_click)
        
    def push_undo(self):
        if self.filtered_df is not None:
            self.undo_stack.append((
                self.filtered_df.copy(deep=True),
                list(self.filters)
            ))
            self.redo_stack.clear()
            self.auto_backup()  # Otomatik yedekle

    def auto_backup(self):
        backup_data = {
            "df": self.df.copy(deep=True) if self.df is not None else None,
            "filtered_df": self.filtered_df.copy(deep=True) if self.filtered_df is not None else None,
            "filters": list(self.filters)
        }
        try:
            with open("auto_backup.pkl", "wb") as f:
                pickle.dump(backup_data, f)
        except Exception as e:
            print("Yedek alınamadı:", e)

    def restore_backup(self):
        if not os.path.exists("auto_backup.pkl"):
            messagebox.showinfo("Yedek", "Yedek bulunamadı.")
            return
        try:
            with open("auto_backup.pkl", "rb") as f:
                backup_data = pickle.load(f)
            self.df = backup_data.get("df")
            self.filtered_df = backup_data.get("filtered_df")
            self.filters = backup_data.get("filters", [])
            self.update_filters_label()
            self.show_data()
            messagebox.showinfo("Yedek", "Yedek başarıyla geri yüklendi.")
        except Exception as e:
            messagebox.showerror("Yedek", f"Yedek geri yüklenemedi: {e}")

    def undo(self):
        if not self.undo_stack:
            messagebox.showinfo("Geri Al", "Geri alınacak bir adım yok.")
            return
        self.redo_stack.append((
            self.filtered_df.copy(deep=True),
            list(self.filters)
        ))
        df, filters = self.undo_stack.pop()
        self.filtered_df = df
        self.filters = filters
        self.update_filters_label()
        self.show_data()
        self.auto_backup()  # Yedek güncelle

    def redo(self):
        if not self.redo_stack:
            messagebox.showinfo("İleri Al", "İleri alınacak bir adım yok.")
            return
        self.undo_stack.append((
            self.filtered_df.copy(deep=True),
            list(self.filters)
        ))
        df, filters = self.redo_stack.pop()
        self.filtered_df = df
        self.filters = filters
        self.update_filters_label()
        self.show_data()
        self.auto_backup()  # Yedek güncelle

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Excel Dosyası", "*.xlsx"),
                ("CSV Dosyası", "*.csv"),
                ("Text Dosyası", "*.txt"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".csv":
                    self.df = pd.read_csv(file_path)
                elif ext == ".xlsx":
                    self.df = pd.read_excel(file_path)
                elif ext == ".txt":
                    try:
                        self.df = pd.read_csv(file_path, delimiter="\t")
                    except Exception:
                        self.df = pd.read_csv(file_path)
                else:
                    messagebox.showerror("Hata", "Desteklenmeyen dosya formatı.")
                    return
                self.filtered_df = self.df.copy()
                self.filters = []
                self.push_undo()
                self.update_filters_label()
                messagebox.showinfo("Başarılı", "Veri başarıyla yüklendi.")
                self.show_data()
            except Exception as e:
                messagebox.showerror("Hata", f"Veri yüklenirken hata oluştu:\n{e}")
    
    def show_data(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        self.update_table(self.filtered_df)
        
    def show_data_popup(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
    
        popup = tk.Toplevel(self)
        popup.title("Veri Tablosu (Kaydırılabilir)")
        popup.geometry("1200x500")
    
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True)
    
        tree = ttk.Treeview(frame, show="headings")
        tree["columns"] = list(self.filtered_df.columns)
        for col in self.filtered_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
    
        for row in self.filtered_df.itertuples(index=False):
            tree.insert("", tk.END, values=row)
    
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
        tk.Button(popup, text="Kapat", command=popup.destroy).pack(pady=4)

    
    def filter_data(self):
        if self.df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return

        def add_filter():
            col = combo_col.get()
            op = combo_op.get()
            val = entry_val.get()
            if not col or not op or not val:
                messagebox.showwarning("Uyarı", "Tüm alanları doldurun.")
                return
            try:
                self.push_undo()
                self.filters.append((col, op, val))
                self.apply_filters()
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Hata", str(e))

        popup = tk.Toplevel(self)
        popup.title("Veriye Filtre Ekle")
        tk.Label(popup, text="Kolon:").grid(row=0, column=0, padx=5, pady=5)
        combo_col = ttk.Combobox(popup, values=list(self.df.columns), state="readonly")
        combo_col.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(popup, text="Operatör:").grid(row=1, column=0, padx=5, pady=5)
        combo_op = ttk.Combobox(popup, values=["=", ">", "<"], state="readonly")
        combo_op.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(popup, text="Değer:").grid(row=2, column=0, padx=5, pady=5)
        entry_val = tk.Entry(popup)
        entry_val.grid(row=2, column=1, padx=5, pady=5)

        btn_apply = tk.Button(popup, text="Filtreyi Ekle", command=add_filter)
        btn_apply.grid(row=3, column=0, columnspan=2, pady=10)

        if self.filters:
            tk.Label(popup, text="Uygulanan Filtreler:", fg="blue").grid(row=4, column=0, columnspan=2)
            tk.Label(popup, text=self.get_filters_str(), fg="blue").grid(row=5, column=0, columnspan=2)

    def apply_filters(self):
        df = self.df
        for col, op, val in self.filters:
            try:
                series = df[col]
                cmp_val = self._typecast(series, val)
                if op == "=":
                    df = df[series == cmp_val]
                elif op == ">":
                    df = df[series > cmp_val]
                elif op == "<":
                    df = df[series < cmp_val]
            except Exception:
                continue
        self.filtered_df = df
        self.update_filters_label()
        self.show_data()

    def clear_filters(self):
        if self.df is not None:
            self.push_undo()
            self.filters = []
            self.filtered_df = self.df.copy()
            self.update_filters_label()
            self.show_data()

    def get_filters_str(self):
        if not self.filters:
            return "Yok"
        return " AND ".join([f"{col} {op} {val}" for (col, op, val) in self.filters])

    def update_filters_label(self):
        self.lbl_filters.config(text=f"Uygulanan Filtreler: {self.get_filters_str()}")

    def _typecast(self, series, value):
        dtype = series.dtype
        if pd.api.types.is_numeric_dtype(dtype):
            return float(value)
        return value

    def scale_data(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        numeric_cols = list(self.filtered_df.select_dtypes(include='number').columns)
        if not numeric_cols:
            messagebox.showwarning("Uyarı", "Sayısal veri bulunamadı.")
            return

        def apply_scaling():
            method = combo_method.get()
            selected = [numeric_cols[i] for i in listbox.curselection()]
            if not selected:
                messagebox.showwarning("Uyarı", "En az bir kolon seçmelisiniz.")
                return

            scaler = None
            if method == "Min-Max (0-1)":
                scaler = MinMaxScaler()
            elif method == "Standart (Z-Score)":
                scaler = StandardScaler()
            elif method == "Robust":
                scaler = RobustScaler()
            elif method == "MaxAbs":
                scaler = MaxAbsScaler()
            else:
                messagebox.showerror("Hata", "Geçersiz ölçeklendirme türü.")
                return

            try:
                self.push_undo()
                scaled = scaler.fit_transform(self.filtered_df[selected].values)
                scaled_df = self.filtered_df.copy()
                scaled_df.loc[:, selected] = scaled
                self.filtered_df = scaled_df
                messagebox.showinfo("Başarılı", f"Seçili kolonlar ({', '.join(selected)}) {method} ile ölçeklendirildi.")
                self.show_data()
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Hata", f"Ölçeklendirme yapılamadı: {e}")

        popup = tk.Toplevel(self)
        popup.title("Veri Ölçeklendirme")
        tk.Label(popup, text="Ölçeklendirme Türü:").grid(row=0, column=0, padx=8, pady=8, sticky="e")
        combo_method = ttk.Combobox(popup, values=["Min-Max (0-1)", "Standart (Z-Score)", "Robust", "MaxAbs"], state="readonly")
        combo_method.grid(row=0, column=1, padx=8, pady=8, sticky="w")
        combo_method.current(0)

        tk.Label(popup, text="Kolon(lar) (Ctrl ile çoklu seç):").grid(row=1, column=0, padx=8, pady=8, sticky="e")
        listbox = tk.Listbox(popup, selectmode=tk.MULTIPLE, exportselection=0, height=8)
        for col in numeric_cols:
            listbox.insert(tk.END, col)
        listbox.grid(row=1, column=1, padx=8, pady=8, sticky="w")

        btn_apply = tk.Button(popup, text="Uygula", command=apply_scaling)
        btn_apply.grid(row=2, column=0, columnspan=2, pady=12)

    def data_transform(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return

        popup = tk.Toplevel(self)
        popup.title("Veri Dönüştür")
        popup.geometry("420x420")

        tk.Label(popup, text="Kolon(lar):").pack(pady=4)
        cols_listbox = tk.Listbox(popup, selectmode=tk.MULTIPLE, exportselection=0, height=12)
        for col in self.filtered_df.columns:
            cols_listbox.insert(tk.END, col)
        cols_listbox.pack(fill=tk.BOTH, expand=True, padx=8)

        def convert_dtype():
            selected = [cols_listbox.get(i) for i in cols_listbox.curselection()]
            if not selected:
                messagebox.showwarning("Uyarı", "Dönüştürmek için en az bir kolon seçin.")
                return
            dtype = dtype_combo.get()
            try:
                self.push_undo()
                for col in selected:
                    if dtype == "int":
                        self.filtered_df[col] = pd.to_numeric(self.filtered_df[col], errors="raise").astype(int)
                    elif dtype == "float":
                        self.filtered_df[col] = pd.to_numeric(self.filtered_df[col], errors="raise").astype(float)
                    elif dtype == "str":
                        self.filtered_df[col] = self.filtered_df[col].astype(str)
                self.show_data()
                messagebox.showinfo("Başarılı", f"Kolon(lar) {dtype} tipine dönüştürüldü.")
            except Exception as e:
                messagebox.showerror("Hata", f"Tip dönüşümü başarısız: {e}")

        def fill_na():
            selected = [cols_listbox.get(i) for i in cols_listbox.curselection()]
            if not selected:
                messagebox.showwarning("Uyarı", "Doldurmak için en az bir kolon seçin.")
                return
            method = na_combo.get()
            try:
                self.push_undo()
                for col in selected:
                    col_data = self.filtered_df[col]
                    # Sahte boşlukları gerçek NaN'a çevir
                    col_data = col_data.replace(["", " ", "-", "nan", "NaN"], np.nan)
                    # Sadece sayıya çevrilebilenleri kullan
                    numeric_data = pd.to_numeric(col_data, errors="coerce")
                    if method == "Ortalama":
                        fill_val = numeric_data.mean()
                    elif method == "Medyan":
                        fill_val = numeric_data.median()
                    elif method == "Mod":
                        try:
                            fill_val = col_data.mode()[0]
                        except Exception:
                            fill_val = ""
                    elif method == "Sabit Değer":
                        fill_val = simpledialog.askstring("Sabit Değer", f"{col} için doldurulacak değeri girin:")
                    # Sadece gerçek eksikleri doldur
                    self.filtered_df[col] = col_data.where(~col_data.isnull(), fill_val)
                    if method == "Satırı Sil":
                        self.filtered_df = self.filtered_df.dropna(subset=[col])
                self.show_data()
                messagebox.showinfo("Başarılı", "Seçili kolon(lar) için eksik değer işlemi uygulandı.")
            except Exception as e:
                messagebox.showerror("Hata", f"İşlem başarısız: {e}")

        def do_encoding():
            selected = [cols_listbox.get(i) for i in cols_listbox.curselection()]
            if not selected:
                messagebox.showwarning("Uyarı", "En az bir kolon seçin.")
                return
            enc_method = enc_combo.get()
            try:
                self.push_undo()
                for col in selected:
                    if enc_method == "Label":
                        self.filtered_df[col] = self.filtered_df[col].astype("category").cat.codes
                    elif enc_method == "One-Hot":
                        df_oh = pd.get_dummies(self.filtered_df[col], prefix=col, dtype=int)
                        self.filtered_df = pd.concat([self.filtered_df.drop(columns=[col]), df_oh], axis=1)
                self.show_data()
                messagebox.showinfo("Başarılı", "Seçili kolon(lar) için encoding uygulandı.")
            except Exception as e:
                messagebox.showerror("Hata", f"Encoding başarısız: {e}")

        dtype_frame = tk.LabelFrame(popup, text="Tip Dönüştür")
        dtype_frame.pack(fill=tk.X, padx=8, pady=6)
        dtype_combo = ttk.Combobox(dtype_frame, values=["int", "float", "str"], state="readonly", width=8)
        dtype_combo.current(0)
        dtype_combo.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(dtype_frame, text="Dönüştür", command=convert_dtype).pack(side=tk.LEFT, padx=10)

        na_frame = tk.LabelFrame(popup, text="Eksik Değer Doldur")
        na_frame.pack(fill=tk.X, padx=8, pady=6)
        na_combo = ttk.Combobox(na_frame, values=["Ortalama", "Medyan", "Mod", "Sabit Değer", "Satırı Sil"], state="readonly", width=11)
        na_combo.current(0)
        na_combo.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(na_frame, text="Uygula", command=fill_na).pack(side=tk.LEFT, padx=10)

        enc_frame = tk.LabelFrame(popup, text="Kategorik Encoding")
        enc_frame.pack(fill=tk.X, padx=8, pady=6)
        enc_combo = ttk.Combobox(enc_frame, values=["Label", "One-Hot"], state="readonly", width=10)
        enc_combo.current(0)
        enc_combo.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(enc_frame, text="Uygula", command=do_encoding).pack(side=tk.LEFT, padx=10)

    def show_stats(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        stats = self.filtered_df.describe(include='all').T
        stats["null_count"] = self.filtered_df.isnull().sum()
        stats["dtype"] = self.filtered_df.dtypes.astype(str)
        stats_cols = ["dtype", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "null_count", "unique", "top", "freq"]
        stats = stats.reindex(columns=[col for col in stats_cols if col in stats.columns])

        stat_headers_tr = {
            "dtype": "Veri Tipi",
            "count": "Sayı",
            "mean": "Ortalama",
            "std": "Standart Sapma",
            "min": "Minimum",
            "25%": "25. Yüzdelik",
            "50%": "Medyan",
            "75%": "75. Yüzdelik",
            "max": "Maksimum",
            "null_count": "Boş Değer",
            "unique": "Tekil Değer",
            "top": "En Sık Değer",
            "freq": "Frekans"
        }
        all_columns = ["Kolon"] + [stat_headers_tr.get(col, col) for col in stats.columns]

        popup = tk.Toplevel(self)
        popup.title("Veri İstatistikleri")
        popup.geometry("1200x600")
        # İstatistikler
        frame_table = tk.Frame(popup)
        frame_table.pack(side=tk.TOP, fill=tk.X)
        tree = ttk.Treeview(frame_table)
        tree.pack(fill=tk.BOTH, expand=True)
        tree["columns"] = all_columns
        tree["show"] = "headings"
        for col in all_columns:
            tree.heading(col, text=col)
            tree.column(col, width=90 if col != "Kolon" else 150)
        for idx, row in stats.iterrows():
            vals = [idx] + [row.get(c, "") for c in stats.columns]
            tree.insert("", tk.END, values=vals)

        # Sayısal kolonlar için küçük histogram ve boxplotlar
        num_cols = self.filtered_df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            frame_graphs = tk.LabelFrame(popup, text="Sayısal Kolonlar için Mini Grafikler")
            frame_graphs.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=10, pady=10)
            images = []  # Referansı kaybetmemek için liste tut
            for col in num_cols:
                subframe = tk.Frame(frame_graphs)
                subframe.pack(side=tk.LEFT, padx=10)
                tk.Label(subframe, text=col, font=("Arial", 10, "bold")).pack()
                # Histogram
                fig, ax = plt.subplots(figsize=(2.2, 1.6))
                ax.hist(self.filtered_df[col].dropna(), bins=15, color="#337ab7", alpha=0.75)
                ax.set_title("Histogram", fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(labelsize=7)
                plt.tight_layout(pad=0.5)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=200)
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf)
                img = img.resize((120, 85), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                images.append(photo)
                label_hist = tk.Label(subframe, image=photo)
                label_hist.pack(pady=2)
                # Boxplot
                fig2, ax2 = plt.subplots(figsize=(2.2, 0.7))
                ax2.boxplot(self.filtered_df[col].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor="#5cb85c"))
                ax2.set_title("Boxplot", fontsize=8)
                ax2.set_yticks([])
                ax2.tick_params(labelsize=7)
                plt.tight_layout(pad=0.5)
                buf2 = io.BytesIO()
                plt.savefig(buf2, format="png", dpi=200)
                plt.close(fig2)
                buf2.seek(0)
                img2 = Image.open(buf2)
                img2 = img2.resize((120, 45), Image.Resampling.LANCZOS)
                photo2 = ImageTk.PhotoImage(img2)
                images.append(photo2)
                label_box = tk.Label(subframe, image=photo2)
                label_box.pack(pady=2)
            # Görsellerin referanslarını tut
            frame_graphs.images = images

    def show_preview(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        popup = tk.Toplevel(self)
        popup.title("Veri Önizle (İlk ve Son 5 Satır)")
        popup.geometry("1150x350")
        tree = ttk.Treeview(popup)
        tree.pack(fill=tk.BOTH, expand=True)
        preview_df = pd.concat([self.filtered_df.head(5), self.filtered_df.tail(5)])
        tree["columns"] = list(preview_df.columns)
        tree["show"] = "headings"
        for col in preview_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        for _, row in preview_df.iterrows():
            tree.insert("", tk.END, values=list(row))

    def visualize_data(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return

        def plot():
            x_col = combo_x.get()
            y_col = combo_y.get()
            chart_type = combo_chart.get()
            if not x_col or not (y_col or chart_type in ["Histogram", "Pie"]):
                messagebox.showwarning("Uyarı", "Alanları seçin.")
                return
            try:
                if self.plot_win is not None and self.plot_win.winfo_exists():
                    self.plot_win.destroy()
                self.plot_win = tk.Toplevel(self)
                self.plot_win.title("Matplotlib Grafiği")
                self.plot_win.geometry("720x520")

                fig, ax = plt.subplots(figsize=(7, 4.5))
                if chart_type == "Scatter":
                    ax.scatter(self.filtered_df[x_col], self.filtered_df[y_col], alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{x_col} vs {y_col}")
                elif chart_type == "Line":
                    ax.plot(self.filtered_df[x_col], self.filtered_df[y_col], marker='o')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{x_col} vs {y_col} (Çizgi Grafik)")
                elif chart_type == "Bar":
                    ax.bar(self.filtered_df[x_col], self.filtered_df[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{x_col} vs {y_col} (Bar Grafik)")
                elif chart_type == "Histogram":
                    ax.hist(self.filtered_df[x_col], bins=20, alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_title(f"{x_col} (Histogram)")
                elif chart_type == "Pie":
                    counts = self.filtered_df[x_col].value_counts()
                    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
                    ax.set_title(f"{x_col} (Pasta Grafik)")
                else:
                    messagebox.showerror("Hata", "Geçersiz grafik türü.")
                    plt.close(fig)
                    return

                canvas = FigureCanvasTkAgg(fig, master=self.plot_win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                plt.close(fig)
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Hata", str(e))

        popup = tk.Toplevel(self)
        popup.title("Veriyi Görselleştir")

        tk.Label(popup, text="Grafik Türü:").grid(row=0, column=0, padx=5, pady=5)
        chart_types = ["Scatter", "Line", "Bar", "Histogram", "Pie"]
        combo_chart = ttk.Combobox(popup, values=chart_types, state="readonly")
        combo_chart.grid(row=0, column=1, padx=5, pady=5)
        combo_chart.current(0)

        tk.Label(popup, text="X Ekseni:").grid(row=1, column=0, padx=5, pady=5)
        combo_x = ttk.Combobox(popup, values=list(self.filtered_df.columns), state="readonly")
        combo_x.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(popup, text="Y Ekseni:").grid(row=2, column=0, padx=5, pady=5)
        combo_y = ttk.Combobox(popup, values=list(self.filtered_df.columns), state="readonly")
        combo_y.grid(row=2, column=1, padx=5, pady=5)

        def on_chart_type_change(event):
            chart_type = combo_chart.get()
            if chart_type in ["Histogram", "Pie"]:
                combo_y.set("")
                combo_y.config(state="disabled")
            else:
                combo_y.config(state="readonly")

        combo_chart.bind("<<ComboboxSelected>>", on_chart_type_change)

        btn_plot = tk.Button(popup, text="Göster", command=plot)
        btn_plot.grid(row=3, column=0, columnspan=2, pady=10)

    def save_filtered_data(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin ve/veya filtre uygulayın.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV Dosyası", "*.csv"),
                ("Excel Dosyası", "*.xlsx"),
                ("Text Dosyası", "*.txt"),
                ("Tüm Dosyalar", "*.*")
            ],
            title="Veriyi Kaydet"
        )
        if not file_path:
            return
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".csv":
                self.filtered_df.to_csv(file_path, index=False)
            elif ext == ".xlsx":
                self.filtered_df.to_excel(file_path, index=False)
            elif ext == ".txt":
                self.filtered_df.to_csv(file_path, sep="\t", index=False)
            else:
                self.filtered_df.to_csv(file_path, index=False)
            messagebox.showinfo("Başarılı", f"Filtrelenmiş veri '{file_path}' olarak kaydedildi.")
        except Exception as e:
            messagebox.showerror("Hata", f"Veri kaydedilemedi:\n{e}")

    def edit_columns(self):
        if self.filtered_df is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return

        def delete_column():
            selected = [cols_listbox.get(i) for i in cols_listbox.curselection()]
            if not selected:
                messagebox.showwarning("Uyarı", "Silmek için en az bir kolon seçin.")
                return
            if messagebox.askyesno("Emin misiniz?", f"{', '.join(selected)} kolon(lar)ı silinsin mi?"):
                self.push_undo()
                self.filtered_df.drop(columns=selected, inplace=True)
                self.show_data()
                update_cols_listbox()
                messagebox.showinfo("Başarılı", "Kolon(lar) silindi.")

        def rename_column():
            selected = [cols_listbox.get(i) for i in cols_listbox.curselection()]
            if len(selected) != 1:
                messagebox.showwarning("Uyarı", "Yeniden adlandırmak için bir kolon seçin.")
                return
            new_name = simpledialog.askstring("Kolon Adı", f"{selected[0]} için yeni isim:")
            if new_name and new_name not in self.filtered_df.columns:
                self.push_undo()
                self.filtered_df.rename(columns={selected[0]: new_name}, inplace=True)
                self.show_data()
                update_cols_listbox()
                messagebox.showinfo("Başarılı", "Kolon adı değiştirildi.")
            elif new_name:
                messagebox.showwarning("Uyarı", "Bu isimde başka bir kolon var.")

        def add_column():
            new_col = simpledialog.askstring("Yeni Kolon", "Yeni kolon adı:")
            if not new_col:
                return
            try:
                expr = simpledialog.askstring("Formül", "Yeni kolon için bir formül girin (örn: A + B * 2):")
                self.push_undo()
                if expr:
                    local_dict = {col: self.filtered_df[col] for col in self.filtered_df.columns}
                    self.filtered_df[new_col] = eval(expr, {}, local_dict)
                    self.show_data()
                    update_cols_listbox()
                    messagebox.showinfo("Başarılı", "Yeni kolon eklendi.")
                else:
                    self.filtered_df[new_col] = None
                    self.show_data()
                    update_cols_listbox()
                    messagebox.showinfo("Başarılı", "Boş yeni kolon eklendi.")
            except Exception as e:
                messagebox.showerror("Hata", f"Kolon eklenemedi:\n{e}")

        def update_cols_listbox():
            cols_listbox.delete(0, tk.END)
            for col in self.filtered_df.columns:
                cols_listbox.insert(tk.END, col)

        popup = tk.Toplevel(self)
        popup.title("Kolonları Düzenle")
        popup.geometry("350x340")
        tk.Label(popup, text="Kolonlar:").pack(pady=4)
        cols_listbox = tk.Listbox(popup, selectmode=tk.MULTIPLE, exportselection=0, height=12)
        cols_listbox.pack(fill=tk.BOTH, expand=True, padx=8)
        update_cols_listbox()
        btns_frame = tk.Frame(popup)
        btns_frame.pack(pady=8)
        tk.Button(btns_frame, text="🗑 Sil", width=10, command=delete_column).grid(row=0, column=0, padx=4)
        tk.Button(btns_frame, text="✏️ Yeniden Adlandır", width=18, command=rename_column).grid(row=0, column=1, padx=4)
        tk.Button(btns_frame, text="➕ Ekle", width=10, command=add_column).grid(row=0, column=2, padx=4)

    def update_table(self, df):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")
        for row in df.itertuples(index=False):
            self.tree.insert("", tk.END, values=row)

if __name__ == "__main__":
    app = VeriCambazi()
    app.mainloop()