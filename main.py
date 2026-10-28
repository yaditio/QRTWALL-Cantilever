import os
import sys
import io
import math
import shutil
import ctypes

# 1. Preload Windows System32 ICU DLLs before Conda Library/bin so PySide6 Qt6Core loads cleanly
if sys.platform == "win32":
    sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for icu_dll in ("icuuc.dll", "icuin.dll"):
        p = os.path.join(sys32, icu_dll)
        if os.path.exists(p):
            try:
                ctypes.WinDLL(p)
            except Exception:
                pass

    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.exists(lib_bin) and lib_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")

import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QFile, QObject, QRect, QUrl, QMarginsF
from PySide6.QtGui import QPixmap, QImage, QColor, QBrush, QFont, QPdfWriter, QTextDocument, QPageSize, QPageLayout
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGroupBox,
    QDialog,
    QFileDialog,
    QMessageBox,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
    QProgressDialog,
)
from PySide6.QtUiTools import QUiLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(BASE_DIR, "engine.py")
MAIN_UI_PATH = os.path.join(BASE_DIR, "mainwindow.ui")
REPORT_UI_PATH = os.path.join(BASE_DIR, "report_dialog.ui")

# ponytail: in-memory dict cache for SSRM results keyed by args tuple; upgrade to disk cache if persistence across restarts is needed
_SSRM_CACHE = {}


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return lambda *a, **kw: _DummyContext()


class StreamlitShim:
    """Lightweight shim intercepting st.* calls inside engine.py using Qt widget values."""

    def __init__(self, inputs: dict, button_states: dict | None = None):
        self.inputs = inputs
        self.buttons = button_states or {}
        self.session_state = {
            "analysis_run": True,
            "trigger_splash": True,
            "trigger_report_splash": False,
            "trigger_report_dialog": False,
        }
        self.sidebar = self
        self.errors = []

    def set_page_config(self, **kw):
        pass

    def dialog(self, *a, **kw):
        def _dec(fn):
            return fn
        return _dec

    def cache_data(self, fn=None, **kw):
        def _wrap(f):
            def _cached(*args, **kwargs):
                key = (f.__name__, args, tuple(sorted(kwargs.items())))
                if key not in _SSRM_CACHE:
                    _SSRM_CACHE[key] = f(*args, **kwargs)
                return _SSRM_CACHE[key]
            return _cached
        return _wrap(fn) if fn is not None else _wrap

    def tabs(self, names):
        return [_DummyContext() for _ in names]

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_DummyContext() for _ in range(n)]

    def container(self, **kw):
        return _DummyContext()

    def expander(self, *a, **kw):
        return _DummyContext()

    def spinner(self, *a, **kw):
        return _DummyContext()

    def empty(self):
        return _DummyContext()

    def number_input(self, label, value=0.0, **kw):
        return self.inputs.get(label, value)

    def selectbox(self, label, options, index=0, **kw):
        default = options[index] if options else None
        return self.inputs.get(label, default)

    def checkbox(self, label, value=False, **kw):
        return bool(self.inputs.get(label, value))

    def button(self, label, key=None, **kw):
        return bool(self.buttons.get(key or label, False))

    def download_button(self, *a, **kw):
        return False

    def markdown(self, *a, **kw):
        pass

    def subheader(self, *a, **kw):
        pass

    def header(self, *a, **kw):
        pass

    def title(self, *a, **kw):
        pass

    def caption(self, *a, **kw):
        pass

    def info(self, *a, **kw):
        pass

    def success(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass

    def error(self, msg, *a, **kw):
        self.errors.append(str(msg))

    def code(self, *a, **kw):
        pass

    def latex(self, *a, **kw):
        pass

    def metric(self, *a, **kw):
        pass

    def image(self, *a, **kw):
        pass

    def pyplot(self, fig=None, *a, **kw):
        target = fig if fig is not None else plt.gcf()
        if target is not None:
            try:
                buf = io.BytesIO()
                target.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0.2)
                target._cached_png_bytes = buf.getvalue()
            except Exception:
                pass

    def table(self, *a, **kw):
        pass

    def dataframe(self, *a, **kw):
        pass

    def stop(self):
        raise RuntimeError("st.stop() called")


class UIProxy:
    def __init__(self, root: QObject):
        self._root = root

    def __getattr__(self, name: str):
        w = self._root.findChild(QObject, name)
        if w is not None:
            setattr(self, name, w)
            return w
        raise AttributeError(f"Widget '{name}' not found in UI")


def load_ui(ui_path: str):
    loader = QUiLoader()
    qf = QFile(ui_path)
    if not qf.open(QFile.ReadOnly):
        raise RuntimeError(f"Cannot open UI file: {ui_path}")
    w = loader.load(qf)
    qf.close()
    if w is None:
        raise RuntimeError(f"QUiLoader failed on {ui_path}: {loader.errorString()}")
    return w


def _crop_transparent_margins(img: QImage, pad: int = 24) -> QImage:
    if img.isNull():
        return img
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 5)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)
    return img.copy(QRect(x1, y1, x2 - x1, y2 - y1))


def drawing_to_pixmap(d_obj) -> QPixmap | None:
    if d_obj is None:
        return None
    try:
        d_obj.set_render_size(1600, 1200)
        png_bytes = d_obj.rasterize().png_data
        img = QImage()
        if img.loadFromData(png_bytes):
            cropped = _crop_transparent_margins(img, pad=28)
            return QPixmap.fromImage(cropped)
    except Exception:
        pass
    try:
        svg_bytes = d_obj.as_svg().encode("utf-8")
        pix = QPixmap()
        if pix.loadFromData(svg_bytes):
            return pix
    except Exception:
        pass
    return None


def fig_to_pixmap(fig, dpi: int = 140) -> QPixmap | None:
    if fig is None:
        return None
    try:
        raw = getattr(fig, "_cached_png_bytes", None)
        if not raw:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.2)
            raw = buf.getvalue()
        pix = QPixmap()
        if pix.loadFromData(raw):
            return pix
    except Exception:
        pass
    return None


def set_label_pixmap(lbl: QLabel, pix: QPixmap | None, fallback_text: str = "Not Available", max_w: int = 920, max_h: int = 580):
    if pix is None or pix.isNull():
        lbl.setPixmap(QPixmap())
        lbl.setText(fallback_text)
        return
    lbl.setText("")
    scaled = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    lbl.setMinimumHeight(scaled.height() + 16)
    lbl.setPixmap(scaled)


def populate_table_from_df(tbl, df: pd.DataFrame, status_col: str | None = None):
    tbl.clear()
    if df is None or df.empty:
        tbl.setRowCount(0)
        tbl.setColumnCount(0)
        return
    cols = list(df.columns)
    tbl.setColumnCount(len(cols))
    tbl.setHorizontalHeaderLabels([str(c) for c in cols])
    tbl.setRowCount(len(df))
    for r_idx in range(len(df)):
        for c_idx, col_name in enumerate(cols):
            val = df.iloc[r_idx, c_idx]
            text = "" if (isinstance(val, float) and math.isnan(val)) else (f"{val:.3f}" if isinstance(val, float) else str(val))
            item = QTableWidgetItem(text)
            if status_col and col_name == status_col:
                up = text.upper()
                font = QFont()
                font.setBold(True)
                item.setFont(font)
                if "PASS" in up:
                    item.setForeground(QBrush(QColor("#16a34a")))
                elif "FAIL" in up:
                    item.setForeground(QBrush(QColor("#dc2626")))
                else:
                    item.setForeground(QBrush(QColor("#2563eb")))
            tbl.setItem(r_idx, c_idx, item)
    tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


class QRTWallApp:
    def __init__(self):
        self.window: QMainWindow = load_ui(MAIN_UI_PATH)
        self.ui = UIProxy(self.window)
        self.ns: dict | None = None
        self._splash_dlg: QProgressDialog | None = None
        self.ui.btn_generate_report.setEnabled(False)
        self._connect_signals()
        self._sync_sidebar_states()
        self._update_result_tabs_visibility(False)

    def _update_result_tabs_visibility(self, has_run: bool, ns: dict | None = None):
        tw = self.ui.resultsTabWidget
        pile_tabs = {"tab_res_pile_analysis", "tab_res_pile_cross_section", "tab_res_pile_verification"}
        pile_on = bool(ns and ns.get("enable_pile"))
        ssrm_on = bool(ns and ns.get("ssrm_results") is not None)

        for i in range(tw.count()):
            name = tw.widget(i).objectName()
            if name == "tab_res_welcome":
                tw.setTabVisible(i, not has_run)
            elif not has_run:
                tw.setTabVisible(i, False)
            elif name in pile_tabs:
                tw.setTabVisible(i, pile_on)
            elif name == "tab_res_ssrm":
                tw.setTabVisible(i, ssrm_on)
            else:
                tw.setTabVisible(i, True)

        if has_run:
            bear_on = bool(ns and ns.get("enable_bearing"))
            self.ui.card_kpi_bearing.setVisible(bear_on)
            self.ui.card_kpi_consol_settle.setVisible(bear_on)
            self.ui.card_kpi_pile_settle.setVisible(pile_on)
            tw.setCurrentWidget(self.ui.tab_res_summary_kpis)
        else:
            tw.setCurrentWidget(self.ui.tab_res_welcome)

    def _connect_signals(self):
        self.ui.chk_include_shear_key.toggled.connect(self._sync_sidebar_states)
        self.ui.combo_surcharge_type.currentTextChanged.connect(self._sync_sidebar_states)
        self.ui.chk_enable_bearing.toggled.connect(self._sync_sidebar_states)
        self.ui.combo_bearing_soil_type.currentTextChanged.connect(self._sync_sidebar_states)
        self.ui.combo_consol_type.currentTextChanged.connect(self._sync_sidebar_states)
        self.ui.chk_enable_pile.toggled.connect(self._sync_sidebar_states)
        self.ui.combo_pile_shape.currentTextChanged.connect(self._sync_sidebar_states)
        self.ui.combo_pile_material.currentTextChanged.connect(self._sync_sidebar_states)
        self.ui.combo_pile_soil_type.currentTextChanged.connect(self._sync_sidebar_states)

        for cb in (
            self.ui.combo_stem_rebar_dia,
            self.ui.combo_stem_sidebar_dia,
            self.ui.combo_ftg_rebar_dia,
            self.ui.combo_ftg_sidebar_dia,
        ):
            cb.currentTextChanged.connect(self._update_rebar_areas)

        self.ui.btn_run_analysis.clicked.connect(self.run_analysis)
        self.ui.btn_generate_report.clicked.connect(self.open_report_dialog)
        self.ui.combo_selected_load_plot.currentTextChanged.connect(self._refresh_load_plot)

        for grp_name in (
            "grp_fem_mesh_expander",
            "grp_stability_equations",
            "grp_load_summary_tables",
            "grp_analytical_load_calcs",
            "grp_bottom_node_disp",
            "grp_concrete_moment_calcs",
            "grp_rebar_cross_sections",
        ):
            grp: QGroupBox = getattr(self.ui, grp_name, None)
            if grp is not None and grp.isCheckable():
                grp.setStyleSheet(
                    "QGroupBox { font-weight: 600; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 6px; margin-top: 10px; padding-top: 14px; } "
                    "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
                )
                grp.toggled.connect(lambda checked, g=grp: self._toggle_expander(g, checked))
                self._toggle_expander(grp, grp.isChecked())

    @staticmethod
    def _toggle_expander(grp: QGroupBox, checked: bool):
        for child in grp.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            child.setVisible(checked)
            child.setEnabled(True)
        grp.setFlat(not checked)
        grp.setMaximumHeight(16777215 if checked else 36)

    def _update_rebar_areas(self):
        def _area(cb):
            d = float(cb.currentText() or 0)
            return math.pi * (d ** 2) / 4.0

        self.ui.lbl_stem_bar_area.setText(f"Calculated Stem Main Bar Area: {_area(self.ui.combo_stem_rebar_dia):.2f} mm²")
        self.ui.lbl_stem_sidebar_area.setText(f"Calculated Stem Side Bar Area: {_area(self.ui.combo_stem_sidebar_dia):.2f} mm²")
        self.ui.lbl_ftg_bar_area.setText(f"Calculated Footing Main Bar Area: {_area(self.ui.combo_ftg_rebar_dia):.2f} mm²")
        self.ui.lbl_ftg_sidebar_area.setText(f"Calculated Footing Side Bar Area: {_area(self.ui.combo_ftg_sidebar_dia):.2f} mm²")

    def _sync_sidebar_states(self):
        self.ui.grp_shear_key.setEnabled(self.ui.chk_include_shear_key.isChecked())

        is_strip = self.ui.combo_surcharge_type.currentText() == "Strip Load"
        self.ui.spin_width_surcharge.setEnabled(is_strip)
        self.ui.spin_offset_surcharge.setEnabled(is_strip)

        bear_on = self.ui.chk_enable_bearing.isChecked()
        self.ui.grp_bearing_params.setEnabled(bear_on)
        self.ui.grp_consol_params.setEnabled(bear_on)
        is_clay_bear = self.ui.combo_bearing_soil_type.currentText() == "Clay (Undrained)"
        self.ui.spin_su_val.setEnabled(bear_on and is_clay_bear)
        is_oc = self.ui.combo_consol_type.currentText() == "Overconsolidated (OC)"
        self.ui.spin_Cr_consol.setEnabled(bear_on and is_oc)
        self.ui.spin_pc_consol.setEnabled(bear_on and is_oc)

        pile_on = self.ui.chk_enable_pile.isChecked()
        self.ui.grp_pile_geom.setEnabled(pile_on)
        self.ui.grp_pile_soil.setEnabled(pile_on)
        self.ui.grp_pile_fs.setEnabled(pile_on)

        is_circle = self.ui.combo_pile_shape.currentText() == "Circle"
        self.ui.spin_diameter_pile.setEnabled(pile_on and is_circle)
        self.ui.spin_width_x_pile.setEnabled(pile_on and not is_circle)
        self.ui.spin_width_y_pile.setEnabled(pile_on and not is_circle)

        is_conc_pile = self.ui.combo_pile_material.currentText() == "Concrete"
        self.ui.spin_n_rebar_pile.setEnabled(pile_on and is_conc_pile)
        self.ui.combo_rebar_dia_pile.setEnabled(pile_on and is_conc_pile)
        self.ui.spin_cover_pile.setEnabled(pile_on and is_conc_pile)

        is_sand_pile = self.ui.combo_pile_soil_type.currentText() == "Sand (Drained)"
        self.ui.spin_phi_pile_soil.setEnabled(pile_on and is_sand_pile)
        self.ui.spin_cohesion_pile_soil.setEnabled(pile_on and is_sand_pile)
        self.ui.spin_delta_pile_axial.setEnabled(pile_on and is_sand_pile)
        self.ui.spin_su_pile_soil.setEnabled(pile_on and not is_sand_pile)
        self.ui.spin_eps50_pile_soil.setEnabled(pile_on and not is_sand_pile)
        self.ui.spin_alpha_pile_axial.setEnabled(pile_on and not is_sand_pile)

        self._update_rebar_areas()

    def collect_inputs(self) -> dict:
        u = self.ui
        return {
            "Out-of-plane thickness (m)": u.spin_t.value(),
            "Wall height (m)": u.spin_Hw.value(),
            "Toe length (m)": u.spin_toe.value(),
            "Heel length (m)": u.spin_heel.value(),
            "Top wall thickness (m)": u.spin_top_wall.value(),
            "Bottom wall thickness (m)": u.spin_bot_wall.value(),
            "Taper Direction": u.combo_taper_direction.currentText(),
            "Footing thickness (m)": u.spin_h_ftg.value(),
            "Include Shear Key": u.chk_include_shear_key.isChecked(),
            "Distance from toe edge (m)": u.spin_shear_key_distance.value(),
            "Shear key width (m)": u.spin_shear_key_width.value(),
            "Shear key thickness (m)": u.spin_shear_key_thickness.value(),
            "Dry soil unit weight (kN/m3)": u.spin_gamma_soil_dry.value(),
            "Wet soil unit weight (kN/m3)": u.spin_gamma_soil_wet.value(),
            "Friction angle (deg)": int(u.spin_phi.value()),
            "Surcharge Type": u.combo_surcharge_type.currentText(),
            "Surcharge load q (kN/m2)": u.spin_q.value(),
            "Surcharge width B (m)": u.spin_width_surcharge.value(),
            "Surcharge offset from wall face a (m)": u.spin_offset_surcharge.value(),
            "Soil height above heel (m)": u.spin_h_soil.value(),
            "Soil above toe (m)": u.spin_h_soil_toe.value(),
            "Soil cohesion for slope stability (kPa)": u.spin_c_soil.value(),
            "Water unit weight (kN/m3)": u.spin_gamma_w.value(),
            "Water height behind wall (m)": u.spin_Hwtr.value(),
            "Water height in front of wall (m)": u.spin_Hwtr_front.value(),
            "Soil Young's Modulus Es (kPa)": u.spin_Es_soil.value(),
            "Soil Poisson's Ratio ν": u.spin_nu_soil.value(),
            "Peak Ground Acceleration (g)": u.spin_PGA.value(),
            "Seismic Load Factor": u.spin_FPGA.value(),
            "Unit weight of concrete (kN/m3)": u.spin_gamma_c.value(),
            "Concrete cohesion (kPa)": u.spin_c_concrete.value(),
            "Comp. strength f'c (MPa)": u.spin_fc.value(),
            "Elastic Modulus Ec (MPa)": u.spin_Ec.value(),
            "Flexural tensile strength (MPa)": u.spin_ft_tensile.value(),
            "Yield strength fsy (MPa)": u.spin_fy.value(),
            "Elastic Modulus Es (MPa)": u.spin_Es.value(),
            "Stem main bar diameter (mm)": int(u.combo_stem_rebar_dia.currentText()),
            "Stem bar spacing x (mm)": u.spin_stem_spacing_x.value(),
            "Stem side bar diameter (mm)": int(u.combo_stem_sidebar_dia.currentText()),
            "Stem side bar spacing y (mm)": u.spin_stem_spacing_y.value(),
            "Stem concrete cover (mm)": u.spin_stem_cover.value(),
            "Footing main bar diameter (mm)": int(u.combo_ftg_rebar_dia.currentText()),
            "Footing bar spacing x (mm)": u.spin_ftg_spacing_x.value(),
            "Footing side bar diameter (mm)": int(u.combo_ftg_sidebar_dia.currentText()),
            "Footing side bar spacing y (mm)": u.spin_ftg_spacing_y.value(),
            "Footing concrete cover (mm)": u.spin_ftg_cover.value(),
            "Enable Bearing Capacity Analysis": u.chk_enable_bearing.isChecked(),
            "Soil Type for Bearing": u.combo_bearing_soil_type.currentText(),
            "Undrained shear strength su (kPa)": u.spin_su_val.value(),
            "Bearing Capacity Safety Factor (FS)": u.spin_FS_bearing.value(),
            "Consolidation Type": u.combo_consol_type.currentText(),
            "Compressible Layer Thickness H₀ (m)": u.spin_H0_consol.value(),
            "Initial Void Ratio e₀": u.spin_e0_consol.value(),
            "Compression Index Cc": u.spin_Cc_consol.value(),
            "Recompression Index Cr": u.spin_Cr_consol.value(),
            "Preconsolidation Pressure p'c (kPa)": u.spin_pc_consol.value(),
            "Enable Pile Foundation Analysis": u.chk_enable_pile.isChecked(),
            "Pile Shape": u.combo_pile_shape.currentText(),
            "Pile Material": u.combo_pile_material.currentText(),
            "Offset from Edge to Pile Centroid (m)": u.spin_pile_offset.value(),
            "Pile Length (m)": u.spin_L_pile.value(),
            "Pile Diameter (m)": u.spin_diameter_pile.value(),
            "Pile Width in Bending direction dx (m)": u.spin_width_x_pile.value(),
            "Pile Width perpendicular to Bending dy (m)": u.spin_width_y_pile.value(),
            "Number of Pile Rebars": int(u.spin_n_rebar_pile.value()),
            "Pile Rebar Diameter (mm)": int(u.combo_rebar_dia_pile.currentText()),
            "Pile Concrete Cover (mm)": u.spin_cover_pile.value(),
            "Soil Type for Piles": u.combo_pile_soil_type.currentText(),
            "Pile Soil Friction Angle (deg)": u.spin_phi_pile_soil.value(),
            "Pile Soil Cohesion (kPa)": u.spin_cohesion_pile_soil.value(),
            "Pile Soil Undrained Shear Strength su (kPa)": u.spin_su_pile_soil.value(),
            "Pile Soil Unit Weight (kN/m3)": u.spin_gamma_pile_soil.value(),
            "Pile Soil Strain at 50% max stress eps50": u.spin_eps50_pile_soil.value(),
            "Axial Safety Factor (piles)": u.spin_FS_pile_axial.value(),
            "Lateral Safety Factor (piles)": u.spin_FS_pile_lateral.value(),
            "OpenPile Loading Type": u.combo_pile_loading_type.currentText(),
            "Pile-Soil Interface Friction Angle δ (deg)": u.spin_delta_pile_axial.value(),
            "Adhesion Factor α (clay)": u.spin_alpha_pile_axial.value(),
            "Select Load Type to Plot": u.combo_selected_load_plot.currentText(),
        }

    def _start_splash(self, win_title: str, init_title: str, init_desc: str):
        dlg = QProgressDialog(self.window)
        dlg.setWindowTitle(win_title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setWindowFlags((dlg.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowTitleHint) & ~Qt.WindowCloseButtonHint & ~Qt.WindowContextHelpButtonHint)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(420)
        dlg.setRange(0, 100)
        dlg.setStyleSheet("""
            QProgressDialog { background-color: #ffffff; }
            QLabel { color: #162447; font-size: 12px; padding: 6px; }
            QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; text-align: center; height: 20px; font-weight: bold; color: #0f172a; background: #f1f5f9; }
            QProgressBar::chunk { background-color: #1f4068; border-radius: 5px; }
        """)
        self._splash_dlg = dlg
        self._on_progress(5, init_title, init_desc)
        dlg.show()
        QApplication.processEvents()

    def _stop_splash(self):
        if self._splash_dlg is not None:
            self._splash_dlg.close()
            self._splash_dlg.deleteLater()
            self._splash_dlg = None

    def _on_progress(self, pct, title_text, desc_text):
        if self._splash_dlg is not None:
            self._splash_dlg.setLabelText(f"{title_text}\n{desc_text}")
            self._splash_dlg.setValue(int(pct))
        self.ui.statusbar.showMessage(f"[{pct}%] {title_text} — {desc_text}")
        QApplication.processEvents()

    def run_analysis(self):
        self.ui.btn_run_analysis.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._start_splash("🧱 Running Analysis", "Initializing Solver", "Preparing geometry and input parameters...")

        try:
            inputs = self.collect_inputs()
            shim = StreamlitShim(inputs, {"btn_run_analysis": True})
            with open(ENGINE_PATH, "r", encoding="utf-8") as f:
                code = f.read()

            ns = {
                "__name__": "__main__",
                "__file__": ENGINE_PATH,
                "st": shim,
                "_qt_progress_hook": self._on_progress,
            }
            old_cwd = os.getcwd()
            os.chdir(BASE_DIR)
            try:
                exec(compile(code, ENGINE_PATH, "exec"), ns)
            finally:
                os.chdir(old_cwd)

            self.ns = ns
            self._on_progress(95, "Rendering Design Dashboard", "Populating result tabs, plots, and tables...")
            self._populate_all_tabs(ns)
            self._update_result_tabs_visibility(True, ns)
            self._on_progress(100, "Analysis Complete", "All checks and figures ready.")
            self.ui.btn_generate_report.setEnabled(True)
            self.ui.lbl_sidebar_info.setText("✅ Analysis complete! Switch tabs on the right or click '📄 Generate Report'.")
            self.ui.statusbar.showMessage("Analysis completed successfully.", 8000)
        except Exception as e:
            self._stop_splash()
            QMessageBox.critical(self.window, "Analysis Error", f"An error occurred during analysis:\n{e}")
            self.ui.statusbar.showMessage(f"Analysis failed: {e}", 8000)
        finally:
            self._stop_splash()
            QApplication.restoreOverrideCursor()
            self.ui.btn_run_analysis.setEnabled(True)

    @staticmethod
    def _set_kpi(val_lbl: QLabel, delta_lbl: QLabel, val_str: str, delta_str: str, is_ok: bool | None = True):
        val_lbl.setText(val_str)
        delta_lbl.setText(delta_str)
        if is_ok is True:
            color = "#16a34a"
        elif is_ok is False:
            color = "#dc2626"
        else:
            color = "#64748b"
        delta_lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600; border: none;")

    def _populate_all_tabs(self, ns: dict):
        u = self.ui

        # --- Tab 1: Resumed Parameters & KPIs ---
        fs_ot = ns.get("FS_ot_global")
        if fs_ot is not None:
            self._set_kpi(u.lbl_kpi_ot_global_val, u.lbl_kpi_ot_global_delta,
                          f"{fs_ot:.3f}" if fs_ot != float("inf") else "∞",
                          "PASS (Min 1.5)" if fs_ot >= 1.5 else "FAIL (Min 1.5)", fs_ot >= 1.5)
        else:
            self._set_kpi(u.lbl_kpi_ot_global_val, u.lbl_kpi_ot_global_delta, "N/A", "-", None)

        fs_sl = ns.get("FS_slide_global")
        if fs_sl is not None:
            self._set_kpi(u.lbl_kpi_slide_global_val, u.lbl_kpi_slide_global_delta,
                          f"{fs_sl:.3f}" if fs_sl != float("inf") else "∞",
                          "PASS (Min 1.5)" if fs_sl >= 1.5 else "FAIL (Min 1.5)", fs_sl >= 1.5)
        else:
            self._set_kpi(u.lbl_kpi_slide_global_val, u.lbl_kpi_slide_global_delta, "N/A", "-", None)

        fs_st = ns.get("FS_ot_stem")
        if fs_st is not None:
            self._set_kpi(u.lbl_kpi_ot_stem_val, u.lbl_kpi_ot_stem_delta,
                          f"{fs_st:.3f}" if fs_st != float("inf") else "∞",
                          "PASS (Min 1.5)" if fs_st >= 1.5 else "FAIL (Min 1.5)", fs_st >= 1.5)
        else:
            self._set_kpi(u.lbl_kpi_ot_stem_val, u.lbl_kpi_ot_stem_delta, "N/A", "-", None)

        if ns.get("enable_bearing"):
            fs_b = ns.get("FS_actual_bearing", 0.0)
            req_b = ns.get("FS_bearing", 3.0)
            b_pass = bool(ns.get("bearing_pass", False))
            b_str = f"{fs_b:.3f}" if fs_b != float("inf") else "∞"
            self._set_kpi(u.lbl_kpi_bearing_val, u.lbl_kpi_bearing_delta,
                          b_str, f"{'PASS' if b_pass else 'FAIL'} (Min {req_b:.2f})", b_pass)
        else:
            self._set_kpi(u.lbl_kpi_bearing_val, u.lbl_kpi_bearing_delta, "N/A", "Not Enabled", None)

        dx_mm = ns.get("max_disp_x", 0.0) * 1000.0
        dy_mm = ns.get("max_disp_y", 0.0) * 1000.0
        self._set_kpi(u.lbl_kpi_disp_val, u.lbl_kpi_disp_delta, f"{dx_mm:.2f} / {dy_mm:.2f} mm", "FEM Peak", None)

        c_pass = bool(ns.get("concrete_pass", False))
        self._set_kpi(u.lbl_kpi_concrete_val, u.lbl_kpi_concrete_delta,
                      "PASS" if c_pass else "FAIL", "Design Safe" if c_pass else "Overstressed", c_pass)

        diff_mm = ns.get("diff_settlement_mm", 0.0)
        diff_ok = bool(ns.get("diff_settlement_pass", True))
        self._set_kpi(u.lbl_kpi_diff_settle_val, u.lbl_kpi_diff_settle_delta,
                      f"{diff_mm:.2f} mm", "PASS (≤50mm)" if diff_ok else "FAIL (≤50mm)", diff_ok)

        rot_deg = ns.get("rotation_deg", 0.0)
        self._set_kpi(u.lbl_kpi_rotation_val, u.lbl_kpi_rotation_delta, f"{rot_deg:.4f}°", "Toe vs Heel", None)

        if ns.get("enable_bearing"):
            c_mm = ns.get("consol_settlement_mm", 0.0)
            c_allow = ns.get("consol_allow_mm", 150.0)
            c_ok = bool(ns.get("consol_pass", True))
            self._set_kpi(u.lbl_kpi_consol_settle_val, u.lbl_kpi_consol_settle_delta,
                          f"{c_mm:.2f} mm", f"{'PASS' if c_ok else 'FAIL'} (≤{c_allow:.1f}mm)", c_ok)
        else:
            self._set_kpi(u.lbl_kpi_consol_settle_val, u.lbl_kpi_consol_settle_delta, "N/A", "Not Enabled", None)

        if ns.get("enable_pile"):
            p_settle = max(ns.get("settlement_toe_pile", 0.0), ns.get("settlement_heel_pile", 0.0))
            p_allow = ns.get("allowable_pile_settlement_mm", 8.0)
            p_ok = p_settle <= p_allow
            self._set_kpi(u.lbl_kpi_pile_settle_val, u.lbl_kpi_pile_settle_delta,
                          f"{p_settle:.2f} mm", f"{'PASS' if p_ok else 'FAIL'} (≤{p_allow:.1f}mm)", p_ok)
        else:
            self._set_kpi(u.lbl_kpi_pile_settle_val, u.lbl_kpi_pile_settle_delta, "N/A", "Not Enabled", None)

        # --- Tab 2: Model Geometry & FEM Mesh ---
        set_label_pixmap(u.svg_model_geometry_view, drawing_to_pixmap(ns.get("d")))
        set_label_pixmap(u.plot_fem_mesh_canvas, fig_to_pixmap(ns.get("fig_model")))

        # --- Tab 3: Stem Bending Moment Diagram ---
        set_label_pixmap(u.plot_stem_moment_canvas, fig_to_pixmap(ns.get("fig_m")))

        # --- Tab 4: SNI Geotechnical Geometry Verification ---
        set_label_pixmap(u.svg_sni_verification_view, drawing_to_pixmap(ns.get("d_sni")))
        populate_table_from_df(u.tbl_sni_checks, ns.get("df_checks"), status_col="Status")
        if ns.get("all_sni_ok", False):
            u.lbl_sni_status_alert.setText(
                "🎉 Success: All wall geometry parameters meet the typical proportions recommended by SNI Perencanaan Geoteknik (Gambar 35b)."
            )
            u.lbl_sni_status_alert.setStyleSheet("background-color: #dcfce7; color: #166534; padding: 8px; border-radius: 6px; font-weight: 600;")
        else:
            u.lbl_sni_status_alert.setText(
                "⚠️ Warning: One or more geometry parameters do not satisfy the typical proportions of SNI Perencanaan Geoteknik (Gambar 35b)."
            )
            u.lbl_sni_status_alert.setStyleSheet("background-color: #fef9c3; color: #854d0e; padding: 8px; border-radius: 6px; font-weight: 600;")

        # --- Tab 5: FE Stress Contours ---
        set_label_pixmap(u.plot_stress_xx_canvas, fig_to_pixmap(ns.get("fig_res")), max_w=430, max_h=330)
        set_label_pixmap(u.plot_stress_yy_canvas, fig_to_pixmap(ns.get("fig_res2")), max_w=430, max_h=330)

        # --- Tab 6: Stability Verification Report ---
        u.txt_stability_report.setHtml(self._build_stability_html(ns))
        u.txt_stability_equations.setHtml(self._build_stability_equations_html(ns))

        # --- Tab 7: Applied Loads & Pressure Distributions ---
        self._refresh_load_plot()
        populate_table_from_df(u.tbl_lateral_loads, ns.get("df_lateral"))
        populate_table_from_df(u.tbl_vertical_loads, ns.get("df_vertical"))
        u.txt_analytical_load_calcs.setHtml(self._build_analytical_loads_html(ns))

        # --- Tab 8: FEM Settlement & Displacement Report ---
        b_ids = ns.get("bottom_node_ids", [1, 8])
        diff_badge = "<span style='color:#16a34a;font-weight:bold;'>[PASS]</span>" if diff_ok else "<span style='color:#dc2626;font-weight:bold;'>[FAIL]</span>"
        u.txt_fem_settlement_summary.setHtml(f"""
        <ul style='line-height:1.6;'>
          <li><b>Toe Settlement (Node {b_ids[0]}):</b> <b>{ns.get('settlement_toe_mm', 0.0):.2f} mm</b>
              (X-disp: {ns.get('disp_toe_x', 0.0)*1000:.2f} mm, Y-disp: {ns.get('disp_toe_y', 0.0)*1000:.2f} mm)</li>
          <li><b>Heel Settlement (Node {b_ids[-1]}):</b> <b>{ns.get('settlement_heel_mm', 0.0):.2f} mm</b>
              (X-disp: {ns.get('disp_heel_x', 0.0)*1000:.2f} mm, Y-disp: {ns.get('disp_heel_y', 0.0)*1000:.2f} mm)</li>
          <li><b>Differential Settlement (Δs):</b> <b>{diff_mm:.2f} mm</b>
              (Allowable: <b>{ns.get('allowable_diff_settlement_mm', 50.0):.1f} mm</b>) {diff_badge}</li>
          <li><b>Footing Slab Rotation:</b> <b>{rot_deg:.4f}°</b> ({ns.get('rotation_rad', 0.0):.6f} rad)</li>
        </ul>
        """)
        populate_table_from_df(u.tbl_bottom_node_disp, ns.get("df_bottom_disp"))

        # --- Tab 9: Concrete Strength Verification ---
        df_conc = pd.DataFrame([
            {
                "Component": "Stem Wall (Base)",
                "Bending Type": "Bending major",
                "Actual M* (kN·m)": f"{ns.get('M_actual_stem', 0.0):.2f}",
                "Capacity φMu (kN·m)": f"{ns.get('M_u_stem_allow', 0.0):.2f}",
                "Status": "PASS" if ns.get("stem_pass") else "FAIL",
            },
            {
                "Component": "Footing Toe (Slab)",
                "Bending Type": "Sagging (Tension Bottom)",
                "Actual M* (kN·m)": f"{ns.get('M_actual_toe', 0.0):.2f}",
                "Capacity φMu (kN·m)": f"{ns.get('M_u_ftg_sag_allow', 0.0):.2f}",
                "Status": "PASS" if ns.get("toe_pass") else "FAIL",
            },
            {
                "Component": "Footing Heel (Slab)",
                "Bending Type": "Hogging (Tension Top)",
                "Actual M* (kN·m)": f"{ns.get('M_actual_heel', 0.0):.2f}",
                "Capacity φMu (kN·m)": f"{ns.get('M_u_ftg_hog_allow', 0.0):.2f}",
                "Status": "PASS" if ns.get("heel_pass") else "FAIL",
            },
        ])
        populate_table_from_df(u.tbl_concrete_strength, df_conc, status_col="Status")
        if c_pass:
            u.lbl_concrete_conclusion.setText(
                f"🤖 Conclusion: All key components (Stem Wall, Footing Toe, Footing Heel) satisfy bending capacity checks "
                f"(Main Rebar D{ns.get('stem_rebar_dia')} Stem, D{ns.get('ftg_rebar_dia')} Footing)."
            )
            u.lbl_concrete_conclusion.setStyleSheet("background-color: #dcfce7; color: #166534; padding: 8px; border-radius: 6px; font-weight: 600;")
        else:
            u.lbl_concrete_conclusion.setText(
                "⚠️ Conclusion: Structural verification FAILED due to insufficient bending capacity. Increase rebar diameter, reduce spacing, or increase thickness."
            )
            u.lbl_concrete_conclusion.setStyleSheet("background-color: #fee2e2; color: #991b1b; padding: 8px; border-radius: 6px; font-weight: 600;")

        populate_table_from_df(u.tbl_concrete_applied_loads, self._build_concrete_loads_df(ns))
        u.txt_concrete_moment_calcs.setHtml(self._build_concrete_moment_calcs_html(ns))

        ref_img_path = os.path.join(BASE_DIR, "wall_section_cuts.png")
        ref_pix = QPixmap(ref_img_path) if os.path.exists(ref_img_path) else None
        set_label_pixmap(u.lbl_wall_section_ref_canvas, ref_pix, max_w=340, max_h=260)
        set_label_pixmap(u.plot_stem_sec_canvas, fig_to_pixmap(ns.get("fig_stem_sec")), max_w=420, max_h=300)
        set_label_pixmap(u.plot_ftg_sec_canvas, fig_to_pixmap(ns.get("fig_ftg_sec")), max_w=420, max_h=300)

        # --- Tabs 10, 11, 12: Pile Foundation ---
        if ns.get("enable_pile"):
            p_toe = ns.get("P_toe", 0.0)
            p_heel = ns.get("P_heel", 0.0)
            self._set_kpi(u.lbl_kpi_pile_toe_p_val, u.lbl_kpi_pile_toe_p_delta, f"{p_toe:.2f} kN", "Compression" if p_toe >= 0 else "Tension", p_toe >= 0)
            self._set_kpi(u.lbl_kpi_pile_toe_v_val, u.lbl_kpi_pile_toe_v_delta, f"{ns.get('V_max_toe', 0.0):.2f} kN", "Max Shear", None)
            self._set_kpi(u.lbl_kpi_pile_toe_m_val, u.lbl_kpi_pile_toe_m_delta, f"{ns.get('M_max_toe', 0.0):.2f} kN·m", "Max Moment", None)
            self._set_kpi(u.lbl_kpi_pile_heel_p_val, u.lbl_kpi_pile_heel_p_delta, f"{p_heel:.2f} kN", "Compression" if p_heel >= 0 else "Tension", p_heel >= 0)
            self._set_kpi(u.lbl_kpi_pile_heel_v_val, u.lbl_kpi_pile_heel_v_delta, f"{ns.get('V_max_heel', 0.0):.2f} kN", "Max Shear", None)
            self._set_kpi(u.lbl_kpi_pile_heel_m_val, u.lbl_kpi_pile_heel_m_delta, f"{ns.get('M_max_heel', 0.0):.2f} kN·m", "Max Moment", None)

            set_label_pixmap(u.svg_pile_foundation_view, drawing_to_pixmap(ns.get("d_pile_svg")), max_w=350, max_h=340)
            set_label_pixmap(u.plot_pile_winkler_canvas, fig_to_pixmap(ns.get("fig_winkler")), max_w=380, max_h=340)
            set_label_pixmap(u.plot_pile_interaction_canvas, fig_to_pixmap(ns.get("fig_int")), max_w=350, max_h=340)
            set_label_pixmap(u.plot_pile_sec_canvas, fig_to_pixmap(ns.get("fig_pile_sec")), max_w=420, max_h=320)

            u.txt_pile_sec_info.setHtml(self._build_pile_sec_info_html(ns))
            populate_table_from_df(u.tbl_pile_verification, self._build_pile_verification_df(ns), status_col="Status")
        else:
            for lbl in (u.svg_pile_foundation_view, u.plot_pile_winkler_canvas, u.plot_pile_interaction_canvas, u.plot_pile_sec_canvas):
                set_label_pixmap(lbl, None, "Pile Foundation Analysis Not Enabled")
            u.txt_pile_sec_info.setHtml("<p>Enable <b>Pile Foundation Analysis</b> in the Pile tab to view pile cross-section details.</p>")
            populate_table_from_df(u.tbl_pile_verification, pd.DataFrame())

        # --- Tab 13: Global Stability Analysis (2D SSRM) ---
        ssrm = ns.get("ssrm_results")
        if ssrm:
            fs_ssrm = ssrm.get("FS")
            is_seis = (ns.get("PGA", 0.0) * ns.get("FPGA", 0.0)) > 0
            req_fs = 1.1 if is_seis else 1.5
            fs_str = "> 3.000" if fs_ssrm is None else f"{fs_ssrm:.3f}"
            ssrm_ok = True if fs_ssrm is None else (fs_ssrm >= req_fs)
            self._set_kpi(u.lbl_kpi_ssrm_fs_val, u.lbl_kpi_ssrm_fs_delta, fs_str, f"Req >= {req_fs}", ssrm_ok)
            u.lbl_ssrm_status_badge.setText(f"Status: {'PASS' if ssrm_ok else 'FAIL'}")
            u.lbl_ssrm_status_badge.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {'#16a34a' if ssrm_ok else '#dc2626'};")

            set_label_pixmap(u.plot_ssrm_inputs_canvas, fig_to_pixmap(ns.get("fig_inputs")))
            set_label_pixmap(u.plot_ssrm_mesh_canvas, fig_to_pixmap(ns.get("fig_mesh")))
            set_label_pixmap(u.plot_ssrm_result_canvas, fig_to_pixmap(ns.get("fig_ssrm")))
            populate_table_from_df(u.tbl_ssrm_materials, ns.get("mat_df"))
            u.txt_ssrm_mesh_summary.setHtml(self._build_ssrm_mesh_html(ns, ssrm))

    def _refresh_load_plot(self):
        if not self.ns:
            return
        ns = self.ns
        u = self.ui
        sel = u.combo_selected_load_plot.currentText()
        try:
            gen_fn = ns.get("generate_load_drawing")
            if callable(gen_fn):
                d_load = gen_fn(sel)
                d_load.set_render_size(800, 550)
                set_label_pixmap(u.svg_load_diagram_view, drawing_to_pixmap(d_load))
        except Exception:
            pass

        Ka = ns.get("Ka", 0.0)
        q = ns.get("q", 0.0)
        h_soil = ns.get("h_soil", 0.0)
        Hwtr = ns.get("Hwtr", 0.0)
        gamma_dry = ns.get("gamma_soil_dry", 18.0)
        gamma_wet = ns.get("gamma_soil_wet", 20.0)
        gamma_w = ns.get("gamma_w", 9.81)
        P_pass = ns.get("P_passive", 0.0)
        soil_lat_f = abs(sum(ns.get("Fx_soil_q", {}).values()))
        water_lat_f = abs(sum(ns.get("Fx_water_lat", {}).values()))

        if ns.get("surcharge_type") == "Strip Load" and "stresses_stripload_retainingwall_local" in ns:
            q_bot = ns["stresses_stripload_retainingwall_local"](
                q, ns.get("width_surcharge", 2.0), ns.get("offset_surcharge", 1.5), h_soil, h_soil
            )["delta sigma x [kPa]"]
        else:
            q_bot = Ka * q
        p_bot = Ka * (gamma_dry * max(0.0, h_soil - Hwtr) + gamma_wet * min(h_soil, Hwtr)) + q_bot
        net_soil_f = max(0.0, soil_lat_f - P_pass)
        p_v = ns.get("q_soil", 0.0) + ns.get("avg_q_v", q)

        u.lbl_peak_soil_lat_press.setText(f"{p_bot:.2f} kPa")
        u.lbl_peak_soil_lat_force.setText(f"{net_soil_f:.2f} kN (Resisted -{P_pass:.2f} kN)")
        u.lbl_peak_soil_vert_press.setText(f"{p_v:.2f} kPa")
        u.lbl_peak_soil_vert_force.setText(f"{abs(ns.get('soil_total', 0.0)):.2f} kN")
        u.lbl_peak_hydro_press.setText(f"{gamma_w * Hwtr:.2f} kPa")
        u.lbl_peak_hydro_force.setText(f"{water_lat_f:.2f} kN")
        u.lbl_peak_uplift_press.setText(f"{gamma_w * Hwtr:.2f} kPa")
        u.lbl_peak_uplift_force.setText(f"{ns.get('uplift_total', 0.0):.2f} kN")
        u.lbl_peak_seismic_force.setText(f"{ns.get('F_seismic_pseudo', 0.0):.2f} kN (kh={ns.get('kh', 0.0):.4f})")

    @staticmethod
    def _build_stability_html(ns: dict) -> str:
        def _badge(val, lim):
            if val is None:
                return "<b>N/A</b>"
            ok = (val == float("inf")) or (val >= lim)
            v_s = "∞" if val == float("inf") else f"{val:.3f}"
            col = "#16a34a" if ok else "#dc2626"
            return f"<span style='color:{col};font-weight:bold;'>[{'PASS' if ok else 'FAIL'} ({v_s})]</span>"

        fs_ot = ns.get("FS_ot_global")
        fs_sl = ns.get("FS_slide_global")
        fs_st = ns.get("FS_ot_stem")
        html = f"""
        <ul style='line-height:1.7;'>
          <li><b>Global Overturning Safety (pivot x = 0.0):</b><br/>
              M<sub>driving</sub> = <b>{ns.get('M_ot_global', 0.0):.3f} kN·m</b> &nbsp;|&nbsp;
              M<sub>resisting</sub> = <b>{ns.get('M_res_global', 0.0):.3f} kN·m</b> &nbsp;&rarr;&nbsp;
              {_badge(fs_ot, 1.5)}</li>
          <li><b>Global Sliding Safety:</b><br/>
              R<sub>sliding</sub> = <b>{ns.get('R_slide', 0.0):.3f} kN</b> &nbsp;|&nbsp;
              F<sub>driving,net</sub> = <b>{ns.get('F_drive', 0.0):.3f} kN</b> (P<sub>passive</sub> = <b>{ns.get('P_passive', 0.0):.3f} kN</b>) &nbsp;&rarr;&nbsp;
              {_badge(fs_sl, 1.5)}</li>
          <li><b>Stem Junction Overturning Safety (pivot x = {ns.get('x_pivot_stem', 0.0):.3f} m):</b><br/>
              M<sub>driving</sub> = <b>{ns.get('M_ot_stem', 0.0):.3f} kN·m</b> &nbsp;|&nbsp;
              M<sub>resisting</sub> = <b>{ns.get('M_res_stem', 0.0):.3f} kN·m</b> &nbsp;&rarr;&nbsp;
              {_badge(fs_st, 1.5)}</li>
        """
        if ns.get("enable_bearing"):
            fs_b = ns.get("FS_actual_bearing", 0.0)
            req_b = ns.get("FS_bearing", 3.0)
            c_ok = bool(ns.get("consol_pass", True))
            html += f"""
            <li><b>Bearing Capacity Safety:</b><br/>
                &sigma;<sub>max</sub> = <b>{ns.get('sigma_max', 0.0):.3f} kPa</b> &nbsp;|&nbsp;
                q<sub>allow</sub> = <b>{ns.get('q_allow', 0.0):.3f} kPa</b> (SF = {req_b:.2f}) &nbsp;&rarr;&nbsp;
                {_badge(fs_b, req_b)}</li>
            <li><b>Consolidation Settlement (SNI 8460:2017):</b><br/>
                s<sub>consol</sub> = <b>{ns.get('consol_settlement_mm', 0.0):.2f} mm</b> &nbsp;|&nbsp;
                s<sub>allow</sub> = <b>{ns.get('consol_allow_mm', 0.0):.1f} mm</b> &nbsp;&rarr;&nbsp;
                <span style='color:{'#16a34a' if c_ok else '#dc2626'};font-weight:bold;'>[{'PASS' if c_ok else 'FAIL'}]</span></li>
            """
        html += "</ul>"
        return html

    @staticmethod
    def _build_stability_equations_html(ns: dict) -> str:
        return """
        <h4>Global Overturning Verification</h4>
        <p><code>M_ot = &Sigma; |F_x,i &middot; z_i|</code><br/>
        <code>M_res = W_base&middot;x_base + W_stem&middot;x_stem + W_soil&middot;x_soil - M_uplift</code><br/>
        <code>FS_ot = M_res / M_ot &ge; 1.5</code></p>
        <h4>Global Sliding Verification</h4>
        <p><code>R_slide = W_down &middot; tan(&phi;) + P_passive,key</code><br/>
        <code>F_drive,net = F_active - P_passive</code><br/>
        <code>FS_slide = (R_slide + P_passive) / F_drive,net &ge; 1.5</code></p>
        <h4>Bearing Capacity &amp; Consolidation Settlement</h4>
        <p><code>&sigma;_max = max(q_toe, q_heel)</code>, &nbsp; <code>q_allow = q_ult / FS_bearing</code><br/>
        <code>s_consol = [H_0 / (1 + e_0)] &middot; C_c &middot; log10((&sigma;'_v0 + &Delta;&sigma;'_v) / &sigma;'_v0)</code><br/>
        <code>s_allow = 15 cm + B(cm)/600</code> (SNI 8460:2017)</p>
        """

    @staticmethod
    def _build_analytical_loads_html(ns: dict) -> str:
        phi = ns.get("phi", 45.0)
        Ka = ns.get("Ka", 0.0)
        Kp = math.tan(math.radians(45.0 + phi / 2.0)) ** 2
        h_soil = ns.get("h_soil", 0.0)
        Hw = ns.get("Hw", 0.0)
        h_ftg = ns.get("h_ftg", 0.3)
        h_soil_toe = ns.get("h_soil_toe", 0.0)
        Hwtr = ns.get("Hwtr", 0.0)
        gamma_dry = ns.get("gamma_soil_dry", 18.0)
        gamma_wet = ns.get("gamma_soil_wet", 20.0)
        gamma_w = ns.get("gamma_w", 9.81)
        q = ns.get("q", 0.0)
        t = ns.get("t", 1.0)
        h_act = min(Hw, h_soil)
        h_dry = max(0.0, h_soil - Hwtr)
        h_wet = min(h_soil, Hwtr)
        pa_top = Ka * q if h_soil > 0 else 0.0
        pa_bot = Ka * (gamma_dry * h_dry + gamma_wet * h_wet) + Ka * q if h_soil > 0 else 0.0
        P_act = 0.5 * (pa_top + pa_bot) * h_act * t if h_soil > 0 else 0.0
        pp_top = (Kp * gamma_dry * h_soil_toe) / 2.0
        pp_bot = (Kp * gamma_dry * (h_soil_toe + h_ftg)) / 2.0
        P_pass = ns.get("P_passive", 0.0)
        P_water = 0.5 * gamma_w * (Hwtr ** 2) * t
        return f"""
        <h4>Earth Pressure Coefficients (Rankine, &phi; = {phi:.1f}&deg;)</h4>
        <p><code>K_a = tan&sup2;(45&deg; - &phi;/2) = <b>{Ka:.4f}</b></code><br/>
        <code>K_p = tan&sup2;(45&deg; + &phi;/2) = <b>{Kp:.4f}</b></code> (Ratio K_p / K_a = {Kp/max(Ka,1e-9):.1f})</p>
        <h4>Active Earth Pressure (Backfill Side)</h4>
        <p><code>p_active,top = {pa_top:.2f} kPa</code>, &nbsp; <code>p_active,bot = {pa_bot:.2f} kPa</code><br/>
        <code>P_active = 0.5 &times; ({pa_top:.2f} + {pa_bot:.2f}) &times; {h_act:.2f} &times; {t:.2f} = <b>{P_act:.2f} kN</b></code></p>
        <h4>Mobilized Passive Earth Pressure (Toe Side, FS = 2.0)</h4>
        <p><code>p_passive,top = {pp_top:.2f} kPa</code>, &nbsp; <code>p_passive,bot = {pp_bot:.2f} kPa</code><br/>
        <code>P_passive = <b>{P_pass:.2f} kN</b></code></p>
        <h4>Hydrostatic, Self-Weight, Uplift &amp; Seismic</h4>
        <p><code>P_water = 0.5 &times; {gamma_w:.2f} &times; {Hwtr:.2f}&sup2; &times; {t:.2f} = <b>{P_water:.2f} kN</b></code><br/>
        <code>W_conc = <b>{ns.get('W_conc', 0.0):.3f} kN</b></code> &nbsp;|&nbsp;
        <code>Soil Vertical = <b>{abs(ns.get('soil_total', 0.0)):.3f} kN</b></code> &nbsp;|&nbsp;
        <code>Uplift = <b>{ns.get('uplift_total', 0.0):.3f} kN</b></code><br/>
        <code>Seismic (k_h = {ns.get('kh', 0.0):.4f}): F_seismic = <b>{ns.get('F_seismic_pseudo', 0.0):.3f} kN</b></code></p>
        """

    @staticmethod
    def _build_concrete_loads_df(ns: dict) -> pd.DataFrame:
        t = ns.get("t", 1.0)
        Hw = ns.get("Hw", 3.0)
        Hwtr = ns.get("Hwtr", 0.0)
        h_ftg = ns.get("h_ftg", 0.3)
        toe = ns.get("toe", 0.8)
        bot_wall = ns.get("bot_wall", 0.4)
        B_len = ns.get("ftg", 2.0)
        heel_len = max(0.0, B_len - (toe + bot_wall))

        fx_soil_q = ns.get("Fx_soil_q", {})
        fx_water = ns.get("Fx_water_lat", {})
        fx_total = ns.get("Fx", {})
        wall_nodes = ns.get("wall_nodes", {})

        f_lat_soil = sum(abs(v) for v in fx_soil_q.values())
        f_lat_water = sum(abs(v) for v in fx_water.values())
        f_lat_seis = abs(ns.get("F_seismic_pseudo", 0.0))

        m_lat_soil = sum(abs(v) * max(0.0, float(wall_nodes.get(n, h_ftg)) - h_ftg) for n, v in fx_soil_q.items())
        m_lat_water = sum(abs(v) * max(0.0, float(wall_nodes.get(n, h_ftg)) - h_ftg) for n, v in fx_water.items())
        n_w = max(1, len(wall_nodes))
        f_seis_node = f_lat_seis / n_w if f_lat_seis > 0 else 0.0
        m_lat_seis = sum(f_seis_node * max(0.0, float(z) - h_ftg) for z in wall_nodes.values())
        f_lat_sum = sum(abs(v) for v in fx_total.values()) + f_lat_seis
        m_stem_actual = ns.get("M_actual_stem", 0.0)

        q_t = ns.get("q_t", 0.0)
        q_h = ns.get("q_h", 0.0)
        q_toe_end = ns.get("q_toe_end", 0.0)
        q_heel_start = ns.get("q_heel_start", 0.0)
        w_down_t = ns.get("w_down_t", 0.0)
        M_bearing_t = ns.get("M_bearing_t", 0.0)
        M_down_t = ns.get("M_down_t", 0.0)
        w_down_h = ns.get("w_down_h", 0.0)
        M_down_h = ns.get("M_down_h", 0.0)
        M_bearing_h = ns.get("M_bearing_h", 0.0)

        sigma_soil_q = ns.get("sigma_soil_q", {})
        p_soil_max = max(sigma_soil_q.values()) if sigma_soil_q else 0.0
        sigma_w_dict = ns.get("sigma_w_dict", {})
        p_w_max = max(sigma_w_dict.values()) if sigma_w_dict else 0.0

        return pd.DataFrame([
            {
                "Component": "Stem Wall (Base)",
                "Load Direction": "Lateral (Horizontal)",
                "Load Source / Description": "Active Earth + Surcharge Pressure (Fx_soil+q)",
                "Intensity / Pressure": f"Peak {p_soil_max:.2f} kPa",
                "Span / Lever Arm": f"Hw = {Hw:.2f} m",
                "Force / Moment Effect": f"F = {f_lat_soil:.2f} kN | M_cant = {m_lat_soil:.2f} kN·m",
            },
            {
                "Component": "Stem Wall (Base)",
                "Load Direction": "Lateral (Horizontal)",
                "Load Source / Description": "Hydrostatic Water Pressure (Fx_water)",
                "Intensity / Pressure": f"Peak {p_w_max:.2f} kPa",
                "Span / Lever Arm": f"Hwtr = {Hwtr:.2f} m",
                "Force / Moment Effect": f"F = {f_lat_water:.2f} kN | M_cant = {m_lat_water:.2f} kN·m",
            },
            {
                "Component": "Stem Wall (Base)",
                "Load Direction": "Lateral (Horizontal)",
                "Load Source / Description": "Seismic Pseudo-Static Inertial Load (kh·W)",
                "Intensity / Pressure": f"kh = {ns.get('kh', 0.0):.4f}",
                "Span / Lever Arm": f"Hw = {Hw:.2f} m",
                "Force / Moment Effect": f"F = {f_lat_seis:.2f} kN | M_cant = {m_lat_seis:.2f} kN·m",
            },
            {
                "Component": "Stem Wall (Base)",
                "Load Direction": "Lateral (FEM Section)",
                "Load Source / Description": f"FEM Base Section Integration (Nodes {ns.get('left_node', 12)}–{ns.get('right_node', 13)})",
                "Intensity / Pressure": f"V_lat = {f_lat_sum:.2f} kN",
                "Span / Lever Arm": f"b_base = {bot_wall:.2f} m",
                "Force / Moment Effect": f"M*_stem = {m_stem_actual:.2f} kN·m/m",
            },
            {
                "Component": "Footing Toe (Slab)",
                "Load Direction": "Vertical (Upward)",
                "Load Source / Description": "Soil Bearing Pressure under Toe (q_0 to q_toe)",
                "Intensity / Pressure": f"{q_t:.2f} → {q_toe_end:.2f} kPa",
                "Span / Lever Arm": f"L_toe = {toe:.2f} m",
                "Force / Moment Effect": f"+M_bearing,toe = {M_bearing_t:.2f} kN·m",
            },
            {
                "Component": "Footing Toe (Slab)",
                "Load Direction": "Vertical (Downward)",
                "Load Source / Description": "Footing Self-Weight + Toe Fill (w_down,toe)",
                "Intensity / Pressure": f"w = {w_down_t:.2f} kN/m",
                "Span / Lever Arm": f"L_toe = {toe:.2f} m",
                "Force / Moment Effect": f"-M_down,toe = {M_down_t:.2f} kN·m  (M*_toe = {ns.get('M_actual_toe', 0.0):.2f} kN·m/m)",
            },
            {
                "Component": "Footing Heel (Slab)",
                "Load Direction": "Vertical (Downward)",
                "Load Source / Description": "Footing + Backfill Soil + Surcharge (w_down,heel)",
                "Intensity / Pressure": f"w = {w_down_h:.2f} kN/m",
                "Span / Lever Arm": f"L_heel = {heel_len:.2f} m",
                "Force / Moment Effect": f"+M_down,heel = {M_down_h:.2f} kN·m",
            },
            {
                "Component": "Footing Heel (Slab)",
                "Load Direction": "Vertical (Upward)",
                "Load Source / Description": "Soil Bearing Pressure under Heel (q_heel to q_B)",
                "Intensity / Pressure": f"{q_heel_start:.2f} → {q_h:.2f} kPa",
                "Span / Lever Arm": f"L_heel = {heel_len:.2f} m",
                "Force / Moment Effect": f"-M_bearing,heel = {M_bearing_h:.2f} kN·m  (M*_heel = {ns.get('M_actual_heel', 0.0):.2f} kN·m/m)",
            },
        ])

    @staticmethod
    def _build_concrete_moment_calcs_html(ns: dict) -> str:
        t = ns.get("t", 1.0)
        h_ftg = ns.get("h_ftg", 0.3)
        h_soil_toe = ns.get("h_soil_toe", 0.0)
        h_dry = ns.get("h_dry", 0.0)
        h_wet = ns.get("h_wet", 0.0)
        gamma_c = ns.get("gamma_c", 24.0)
        gamma_dry = ns.get("gamma_soil_dry", 18.0)
        gamma_wet = ns.get("gamma_soil_wet", 20.0)
        q = ns.get("q", 0.0)
        toe = ns.get("toe", 0.8)
        bot_wall = ns.get("bot_wall", 0.4)
        B_len = ns.get("ftg", 2.0)
        heel_len = max(0.0, B_len - (toe + bot_wall))

        left_node = ns.get("left_node", 12)
        right_node = ns.get("right_node", 13)
        node_moments = ns.get("node_moments_dict", {})
        m_base_raw = abs(node_moments.get(left_node, 0.0))
        m_stem = ns.get("M_actual_stem", 0.0)

        fx_soil_q = ns.get("Fx_soil_q", {})
        fx_water = ns.get("Fx_water_lat", {})
        wall_nodes = ns.get("wall_nodes", {})
        f_lat_soil = sum(abs(v) for v in fx_soil_q.values())
        f_lat_water = sum(abs(v) for v in fx_water.values())
        f_lat_seis = abs(ns.get("F_seismic_pseudo", 0.0))

        Rv = ns.get("Rv_force", 0.0)
        M_res = ns.get("M_res_global", 0.0)
        M_ot = ns.get("M_ot_global", 0.0)
        M_net = ns.get("M_net_moment", 0.0)
        x_R = ns.get("x_R_loc", 0.0)
        e_ecc = ns.get("e_ecc", 0.0)

        q_0 = ns.get("q_t", 0.0)
        q_B = ns.get("q_h", 0.0)
        q_toe_end = ns.get("q_toe_end", 0.0)
        q_heel_start = ns.get("q_heel_start", 0.0)

        w_down_t = ns.get("w_down_t", 0.0)
        M_bearing_t = ns.get("M_bearing_t", 0.0)
        M_down_t = ns.get("M_down_t", 0.0)
        m_toe = ns.get("M_actual_toe", 0.0)

        w_down_h = ns.get("w_down_h", 0.0)
        M_bearing_h = ns.get("M_bearing_h", 0.0)
        M_down_h = ns.get("M_down_h", 0.0)
        m_heel = ns.get("M_actual_heel", 0.0)

        return f"""
        <h4>1. Stem Wall (Base) — Actual Bending Moment M*<sub>stem</sub> (from Lateral Loads)</h4>
        <p>Lateral nodal loads applied along the stem height (<code>H_w = {ns.get('Hw', 3.0):.2f} m</code>):<br/>
        <code>&Sigma;|F_x,soil+q| = {f_lat_soil:.2f} kN</code>, &nbsp;
        <code>&Sigma;|F_x,water| = {f_lat_water:.2f} kN</code>, &nbsp;
        <code>&Sigma;|F_x,seismic| = {f_lat_seis:.2f} kN</code><br/>
        FEM base section moment integrated across bottom stem elements at nodes <code>({left_node}, {right_node})</code>:<br/>
        <code>M_base,FEM = -&Sigma; [F_y,ele &times; (x_node - x_centroid,base)] = <b>{m_base_raw:.2f} kN&middot;m</b></code><br/>
        <code><b>M*<sub>stem</sub></b> = |M_base,FEM| / t = {m_base_raw:.2f} / {t:.2f} = <b>{m_stem:.2f} kN&middot;m/m</b></code></p>

        <h4>2. Base Soil Bearing Pressure Distribution (from Global Vertical &amp; Overturning Equilibrium)</h4>
        <p><code>R_v = max(0.1, W_conc + W_soil + W_surcharge - U_uplift) = <b>{Rv:.2f} kN</b></code><br/>
        <code>M_net = M_res,global - M_ot,global = {M_res:.2f} - {M_ot:.2f} = <b>{M_net:.2f} kN&middot;m</b></code><br/>
        <code>x_R = M_net / R_v = {x_R:.3f} m</code>, &nbsp;
        <code>e = B/2 - x_R = {B_len/2.0:.3f} - {x_R:.3f} = <b>{e_ecc:.3f} m</b></code> (Kern limit <code>B/6 = {B_len/6.0:.3f} m</code>)<br/>
        <code>q(0) = <b>{q_0:.2f} kPa</b></code> (Toe tip), &nbsp;
        <code>q(L_toe) = <b>{q_toe_end:.2f} kPa</b></code> (Stem front), &nbsp;
        <code>q(L_toe + b_stem) = <b>{q_heel_start:.2f} kPa</b></code> (Stem back), &nbsp;
        <code>q(B) = <b>{q_B:.2f} kPa</b></code> (Heel tip)</p>

        <h4>3. Footing Toe (Slab) — Actual Bending Moment M*<sub>toe</sub> (Sagging / Tension Bottom)</h4>
        <p>Span <code>L_toe = {toe:.2f} m</code>:<br/>
        Upward bearing moment:
        <code>M_bearing,toe = [ q(0)&middot;L_toe&sup2;/3 + q(L_toe)&middot;L_toe&sup2;/6 ] &times; t = [ {q_0:.2f}&times;{toe:.2f}&sup2;/3 + {q_toe_end:.2f}&times;{toe:.2f}&sup2;/6 ] &times; {t:.2f} = <b>{M_bearing_t:.2f} kN&middot;m</b></code><br/>
        Downward self-weight + toe soil load:
        <code>w_down,toe = (&gamma;_c&middot;h_ftg + &gamma;_dry&middot;h_soil,toe) &times; t = ({gamma_c:.1f}&times;{h_ftg:.2f} + {gamma_dry:.1f}&times;{h_soil_toe:.2f}) &times; {t:.2f} = <b>{w_down_t:.2f} kN/m</b></code><br/>
        <code>M_down,toe = 0.5 &times; w_down,toe &times; L_toe&sup2; = 0.5 &times; {w_down_t:.2f} &times; {toe:.2f}&sup2; = <b>{M_down_t:.2f} kN&middot;m</b></code><br/>
        <code><b>M*<sub>toe</sub></b> = max(0, M_bearing,toe - M_down,toe) / t = max(0, {M_bearing_t:.2f} - {M_down_t:.2f}) / {t:.2f} = <b>{m_toe:.2f} kN&middot;m/m</b></code></p>

        <h4>4. Footing Heel (Slab) — Actual Bending Moment M*<sub>heel</sub> (Hogging / Tension Top)</h4>
        <p>Span <code>L_heel = B - (L_toe + b_stem) = {B_len:.2f} - ({toe:.2f} + {bot_wall:.2f}) = {heel_len:.2f} m</code>:<br/>
        Downward footing + heel soil + surcharge load:
        <code>w_down,heel = [ &gamma;_c&middot;h_ftg + (&gamma;_dry&middot;h_dry + &gamma;_wet&middot;h_wet)&middot;(L_heel/B) + q ] &times; t = <b>{w_down_h:.2f} kN/m</b></code><br/>
        <code>M_down,heel = 0.5 &times; w_down,heel &times; L_heel&sup2; = 0.5 &times; {w_down_h:.2f} &times; {heel_len:.2f}&sup2; = <b>{M_down_h:.2f} kN&middot;m</b></code><br/>
        Upward counteracting bearing moment:
        <code>M_bearing,heel = [ q(L_toe+b_stem)&middot;L_heel&sup2;/6 + q(B)&middot;L_heel&sup2;/3 ] &times; t = [ {q_heel_start:.2f}&times;{heel_len:.2f}&sup2;/6 + {q_B:.2f}&times;{heel_len:.2f}&sup2;/3 ] &times; {t:.2f} = <b>{M_bearing_h:.2f} kN&middot;m</b></code><br/>
        <code><b>M*<sub>heel</sub></b> = max(0, M_down,heel - M_bearing,heel) / t = max(0, {M_down_h:.2f} - {M_bearing_h:.2f}) / {t:.2f} = <b>{m_heel:.2f} kN&middot;m/m</b></code></p>
        """

    @staticmethod
    def _build_pile_sec_info_html(ns: dict) -> str:
        mat = ns.get("pile_material", "Concrete")
        shape = ns.get("pile_shape", "Circle")
        d = ns.get("diameter_pile", 0.4)
        wx = ns.get("width_x_pile", 0.4)
        wy = ns.get("width_y_pile", 0.4)
        cov = ns.get("cover_pile", 50.0)
        n_reb = int(ns.get("n_rebar_pile", 8))
        dia_reb = ns.get("rebar_dia_pile", 25)
        if mat == "Concrete":
            if shape == "Circle":
                return f"""
                <h4>Circular Concrete Pile Section</h4>
                <ul>
                  <li><b>Pile Diameter:</b> {d:.3f} m ({d*1000:.0f} mm)</li>
                  <li><b>Concrete Cover:</b> {cov:.1f} mm</li>
                  <li><b>Reinforcement:</b> {n_reb} main bars of D{dia_reb}</li>
                  <li><b>Rebar Array Radius:</b> {d*1000/2 - cov:.1f} mm</li>
                </ul>
                """
            return f"""
            <h4>Rectangular Concrete Pile Section</h4>
            <ul>
              <li><b>Dimensions (dx &times; dy):</b> {wx:.3f} m &times; {wy:.3f} m</li>
              <li><b>Concrete Cover:</b> {cov:.1f} mm</li>
              <li><b>Reinforcement:</b> 4 corner bars of D{dia_reb}</li>
            </ul>
            """
        return f"<h4>Solid Steel {shape} Pile Section</h4><p>Dimensions: {d:.3f} m</p>"

    @staticmethod
    def _build_pile_verification_df(ns: dict) -> pd.DataFrame:
        p_toe = ns.get("P_toe", 0.0)
        p_heel = ns.get("P_heel", 0.0)
        q_c = ns.get("Q_comp_allow", 0.0)
        q_t = ns.get("Q_tens_allow", 0.0)
        h_al = ns.get("H_allow", 0.0)
        fs_ax = ns.get("FS_pile_axial", 2.5)
        fs_lat = ns.get("FS_pile_lateral", 2.5)
        s_al = ns.get("allowable_pile_settlement_mm", 8.0)
        return pd.DataFrame([
            {
                "Check Item": "Toe Pile Axial (Compression)",
                "Demand (Actual)": f"{p_toe:.2f} kN",
                "Capacity (Allowable)": f"{q_c:.2f} kN",
                "Safety Factor / Limits": f"FS = {fs_ax:.2f}",
                "Status": "PASS" if ns.get("pile_pass_toe_axial") else "FAIL",
            },
            {
                "Check Item": "Toe Pile Lateral Shear",
                "Demand (Actual)": f"{ns.get('V_max_toe', 0.0):.2f} kN",
                "Capacity (Allowable)": f"{h_al:.2f} kN",
                "Safety Factor / Limits": f"FS = {fs_lat:.2f}",
                "Status": "PASS" if ns.get("pile_pass_toe_lateral") else "FAIL",
            },
            {
                "Check Item": "Toe Pile Structural (Bending-Axial)",
                "Demand (Actual)": f"M={ns.get('M_max_toe', 0.0):.1f} kNm, P={p_toe:.1f} kN",
                "Capacity (Allowable)": "Interaction Envelope",
                "Safety Factor / Limits": "-",
                "Status": "PASS" if ns.get("pile_pass_toe_interaction") else "FAIL",
            },
            {
                "Check Item": "Toe Pile Settlement (Axial openpile)",
                "Demand (Actual)": f"{ns.get('settlement_toe_pile', 0.0):.2f} mm",
                "Capacity (Allowable)": f"{s_al:.1f} mm",
                "Safety Factor / Limits": "2% Diameter",
                "Status": "PASS" if ns.get("pile_pass_toe_settlement") else "FAIL",
            },
            {
                "Check Item": f"Heel Pile Axial ({'Compression' if p_heel >= 0 else 'Tension'})",
                "Demand (Actual)": f"{abs(p_heel):.2f} kN",
                "Capacity (Allowable)": f"{q_c if p_heel >= 0 else q_t:.2f} kN",
                "Safety Factor / Limits": f"FS = {fs_ax:.2f}",
                "Status": "PASS" if ns.get("pile_pass_heel_axial") else "FAIL",
            },
            {
                "Check Item": "Heel Pile Lateral Shear",
                "Demand (Actual)": f"{ns.get('V_max_heel', 0.0):.2f} kN",
                "Capacity (Allowable)": f"{h_al:.2f} kN",
                "Safety Factor / Limits": f"FS = {fs_lat:.2f}",
                "Status": "PASS" if ns.get("pile_pass_heel_lateral") else "FAIL",
            },
            {
                "Check Item": "Heel Pile Structural (Bending-Axial)",
                "Demand (Actual)": f"M={ns.get('M_max_heel', 0.0):.1f} kNm, P={p_heel:.1f} kN",
                "Capacity (Allowable)": "Interaction Envelope",
                "Safety Factor / Limits": "-",
                "Status": "PASS" if ns.get("pile_pass_heel_interaction") else "FAIL",
            },
            {
                "Check Item": "Heel Pile Settlement (Axial openpile)",
                "Demand (Actual)": f"{ns.get('settlement_heel_pile', 0.0):.2f} mm",
                "Capacity (Allowable)": f"{s_al:.1f} mm",
                "Safety Factor / Limits": "2% Diameter",
                "Status": "PASS" if ns.get("pile_pass_heel_settlement") else "FAIL",
            },
        ])

    @staticmethod
    def _build_ssrm_mesh_html(ns: dict, ssrm: dict) -> str:
        fd = ssrm.get("fem_data", {})
        n_nodes = len(fd.get("nodes", []))
        n_elems = len(fd.get("elements", []))
        mats = fd.get("element_materials", np.array([]))
        soil_c = int(np.sum(mats == 1)) if len(mats) else 0
        conc_c = int(np.sum(mats == 2)) if len(mats) else 0
        bc = fd.get("bc_type", np.array([]))
        n_fix = int(np.sum(bc == 1)) if len(bc) else 0
        n_rol = int(np.sum(bc == 2)) if len(bc) else 0
        u_arr = fd.get("u")
        u_str = f"{u_arr.max():.2f} kPa" if (u_arr is not None and len(u_arr) and u_arr.max() > 0) else "None (Dry)"
        return f"""
        <h4>Mesh &amp; Domain Parameters</h4>
        <ul>
          <li><b>Total Mesh Nodes:</b> {n_nodes}</li>
          <li><b>Total 2D Elements (tri6):</b> {n_elems}</li>
          <li><b>Soil Zone Elements:</b> {soil_c}</li>
          <li><b>Concrete Wall Elements:</b> {conc_c}</li>
          <li><b>Boundary Constraints:</b> {n_fix} fixed base nodes, {n_rol} roller side nodes</li>
          <li><b>Max Pore Pressure (u_max):</b> {u_str}</li>
        </ul>
        """

    def _generate_pdf_from_docx(self, docx_path: str, pdf_path: str):
        import html as _html
        from docx import Document
        from docx.text.paragraph import Paragraph as DocxParagraph
        from docx.table import Table as DocxTable

        wdoc = Document(docx_path)
        qdoc = QTextDocument()
        writer = QPdfWriter(pdf_path)
        writer.setResolution(96)
        writer.setPageSize(QPageSize(QPageSize.A4))
        writer.setPageMargins(QMarginsF(25.4, 25.4, 25.4, 25.4), QPageLayout.Millimeter)

        parts = [
            "<html><head><style>"
            "body { font-family: Arial, sans-serif; font-size: 10pt; color: #1e293b; line-height: 1.4; }"
            "h1 { color: #1f4068; font-size: 15pt; margin-top: 16px; margin-bottom: 6px; }"
            "h2 { color: #2b580c; font-size: 12pt; margin-top: 12px; margin-bottom: 4px; }"
            "p { margin-top: 4px; margin-bottom: 6px; }"
            "table { border-collapse: collapse; width: 100%; margin-top: 6px; margin-bottom: 10px; }"
            "th { background-color: #e2e8f0; color: #0f172a; font-weight: bold; padding: 5px; border: 1px solid #94a3b8; text-align: left; }"
            "td { padding: 5px; border: 1px solid #cbd5e1; }"
            "</style></head><body>"
        ]

        img_idx = 0
        r_embed_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

        for child in wdoc.element.body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                p = DocxParagraph(child, wdoc)
                style_name = p.style.name if p.style else ""

                # Check for embedded images inside paragraph
                blips = child.xpath(".//*[local-name()='blip']")
                extents = child.xpath(".//*[local-name()='extent']")
                for idx_b, blip in enumerate(blips):
                    rId = blip.get(r_embed_attr)
                    if rId and rId in wdoc.part.related_parts:
                        img_bytes = wdoc.part.related_parts[rId].blob
                        qimg = QImage.fromData(img_bytes)
                        if not qimg.isNull():
                            img_idx += 1
                            res_url = f"docximg://img_{img_idx}.png"
                            qdoc.addResource(QTextDocument.ImageResource, QUrl(res_url), qimg)
                            if idx_b < len(extents):
                                cx = int(extents[idx_b].get("cx", 0))
                                cy = int(extents[idx_b].get("cy", 0))
                                w_px = max(40, min(560, int((cx / 914400.0) * 96)))
                                h_px = max(20, int((cy / 914400.0) * 96))
                                if w_px < int((cx / 914400.0) * 96):
                                    h_px = int(h_px * (w_px / max(1, int((cx / 914400.0) * 96))))
                                parts.append(f'<p><img src="{res_url}" width="{w_px}" height="{h_px}"/></p>')
                            else:
                                parts.append(f'<p><img src="{res_url}"/></p>')

                # Format paragraph text runs
                run_htmls = []
                for run in p.runs:
                    if not run.text:
                        continue
                    t_esc = _html.escape(run.text).replace("\n", "<br/>")
                    if run.bold:
                        t_esc = f"<b>{t_esc}</b>"
                    if run.italic:
                        t_esc = f"<i>{t_esc}</i>"
                    if run.font and run.font.size:
                        pt = int(run.font.size.pt)
                        t_esc = f'<span style="font-size:{pt}pt;">{t_esc}</span>'
                    run_htmls.append(t_esc)

                text_html = "".join(run_htmls)
                if not text_html and p.text:
                    text_html = _html.escape(p.text).replace("\n", "<br/>")

                if text_html:
                    if style_name == "Heading 1":
                        parts.append(f"<h1>{text_html}</h1>")
                    elif style_name == "Heading 2":
                        parts.append(f"<h2>{text_html}</h2>")
                    else:
                        parts.append(f"<p>{text_html}</p>")

            elif tag == "tbl":
                tbl = DocxTable(child, wdoc)
                parts.append('<table width="100%" cellspacing="0" cellpadding="5" border="1">')
                for r_i, row in enumerate(tbl.rows):
                    parts.append("<tr>")
                    cell_tag = "th" if r_i == 0 else "td"
                    for cell in row.cells:
                        c_txt = _html.escape(cell.text.strip()).replace("\n", "<br/>")
                        parts.append(f"<{cell_tag}>{c_txt}</{cell_tag}>")
                    parts.append("</tr>")
                parts.append("</table>")

        parts.append("</body></html>")
        qdoc.setHtml("".join(parts))
        qdoc.print_(writer)

    def open_report_dialog(self):
        if not self.ns:
            QMessageBox.information(self.window, "Run Analysis First", "Please run the analysis before generating a report.")
            return

        ns = self.ns
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._start_splash("📄 Generating Report", "Exporting Diagrams & High-Res Plots", "Exporting geometry, mesh, and SSRM failure surface plots...")

        old_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        os.makedirs("report_temp", exist_ok=True)
        docx_temp = os.path.join(BASE_DIR, "report_temp", "retaining_wall_report.docx")
        pdf_temp = os.path.join(BASE_DIR, "report_temp", "retaining_wall_report.pdf")

        slide_val = ns.get("FS_slide_global") or 0.0
        ot_val = ns.get("FS_ot_global") or 0.0
        summary_html = f"""
        <h3>RT Wall Cantilever — Design Summary Preview</h3>
        <ul>
          <li><b>Wall Height:</b> {ns.get('Hw', 0.0):.3f} m &nbsp;|&nbsp; <b>Footing Width:</b> {ns.get('ftg', 0.0):.3f} m</li>
          <li><b>Backfill Soil Friction Angle:</b> {ns.get('phi', 0.0):.1f}&deg; &nbsp;|&nbsp; <b>Cohesion:</b> {ns.get('c_soil', 0.0):.1f} kPa</li>
          <li><b>Sliding safety check:</b> <b>{'PASS' if slide_val >= 1.5 else 'FAIL'}</b> (FS = {slide_val:.3f})</li>
          <li><b>Overturning safety check:</b> <b>{'PASS' if ot_val >= 1.5 else 'FAIL'}</b> (FS = {ot_val:.3f})</li>
        </ul>
        """

        try:
            self._on_progress(20, "Exporting Diagrams & High-Res Plots", "Exporting geometry, mesh, and SSRM failure surface plots...")
            if callable(ns.get("export_report_assets")):
                ns["export_report_assets"]()
            self._on_progress(55, "Generating Word Document (.docx)", "Building calculation tables, checklist, and embedded figures...")
            if callable(ns.get("generate_docx_report")):
                ns["generate_docx_report"](docx_temp)
            self._on_progress(85, "Compiling PDF Report (.pdf)", "Rendering PDF report with identical DOCX content...")
            if os.path.exists(docx_temp):
                self._generate_pdf_from_docx(docx_temp, pdf_temp)
            self._on_progress(100, "Report Generation Complete", "Opening export dialog...")
        finally:
            os.chdir(old_cwd)
            self._stop_splash()
            QApplication.restoreOverrideCursor()
            self.ui.statusbar.showMessage("Report ready.", 5000)

        dlg: QDialog = load_ui(REPORT_UI_PATH)
        dlg_ui = UIProxy(dlg)
        dlg_ui.txt_design_summary_preview.setHtml(summary_html)

        # Render model preview into pdf_preview_container
        set_label_pixmap(dlg_ui.pdf_preview_container, drawing_to_pixmap(ns.get("d")), f"Report generated at:\n{pdf_temp}")

        def _save_docx():
            dest, _ = QFileDialog.getSaveFileName(dlg, "Save Word Report", "retaining_wall_report.docx", "Word Document (*.docx)")
            if dest and os.path.exists(docx_temp):
                shutil.copyfile(docx_temp, dest)
                QMessageBox.information(dlg, "Saved", f"Saved Word report to:\n{dest}")

        def _save_pdf():
            dest, _ = QFileDialog.getSaveFileName(dlg, "Save PDF Report", "retaining_wall_report.pdf", "PDF Document (*.pdf)")
            if dest and os.path.exists(pdf_temp):
                shutil.copyfile(pdf_temp, dest)
                QMessageBox.information(dlg, "Saved", f"Saved PDF report to:\n{dest}")

        dlg_ui.btn_download_docx.clicked.connect(_save_docx)
        dlg_ui.btn_download_pdf.clicked.connect(_save_pdf)
        dlg.exec()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    ctrl = QRTWallApp()
    ctrl.window.show()
    if "--self-check" in sys.argv:
        ctrl.run_analysis()
        assert ctrl.ns is not None
        assert ctrl.ns.get("FS_ot_global") is not None
        assert ctrl.ns.get("FS_slide_global") is not None
        print("SELF_CHECK_OK:", round(ctrl.ns["FS_ot_global"], 3), round(ctrl.ns["FS_slide_global"], 3))
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
