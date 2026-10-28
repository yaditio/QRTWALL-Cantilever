# 🧱 QRTWALL-Cantilever

**QRTWALL-Cantilever** is a native **Qt (`PySide6`)** desktop port of **RTWALL-Cantilever**, providing all of its cantilever retaining wall analysis, design, and verification functions in a native Qt interface.

---

## ⚠️ Disclaimer & Open Contribution

This application is currently **under active development** and was built with the **assistance of AI**. As an evolving project, there is always room for mistakes, edge-case bugs, and further engineering or UI improvements.

- **Not a Certified Design Tool**: This software is intended for educational, learning, and preliminary engineering exploration. Always verify all results independently against applicable codes and standards (**SNI 8460:2017**, **SNI 2847**, **ACI 318**, **AS 3600**) before making engineering decisions.
- **Free to Fork, Download & Contribute**: Everyone is completely free to download, use, modify, and fork this repository. Contributions of any kind—bug reports, calculation checks, code improvements, documentation, or feature ideas—are warmly welcomed and deeply appreciated!

---

## 🗂️ Project Structure

| File | Role |
|---|---|
| `main.py` | Native `PySide6` desktop application controller (`QRTWallApp`), dynamic `.ui` loader (`QUiLoader`), modal progress splash dialog (`QProgressDialog`), figure/SVG auto-crop rasterizer, and native DOCX/PDF report exporter (`QTextDocument` + `QPdfWriter`). |
| `engine.py` | Computational geotechnical & structural FEM engine (OpenSeesPy 2D `Tri31` mesh, Rankine/Boussinesq pressures, Groundhog bearing & consolidation, ConcreteProperties rebar sections, OpenPile Winkler model, XSLOPE 2D SSRM). |
| `mainwindow.ui` | Qt Designer XML layout defining the left 7-tab input sidebar and right 12-tab dynamic results workspace. |
| `report_dialog.ui` | Qt Designer XML layout for the engineering report preview and export dialog (**DOCX** & **PDF**). |
| `wall_section_cuts.png` | Reference section-cut diagram for the Concrete Strength Verification tab (Stem Base & Footing Slab sections). |

---

## 🖥️ Application Layout & Features

### 1. Left Sidebar — Input Configuration (7 Tabs)
- **Geometry**: Out-of-plane thickness $t$, wall height $H_w$, toe length $L_{\text{toe}}$, heel length $L_{\text{heel}}$, top/bottom stem thickness ($b_{\text{top}}, b_{\text{bot}}$), taper direction (`Toe-facing` / `Heel-facing`), footing thickness $h_{\text{ftg}}$, and optional protruding base **Shear Key** ($x_{\text{sk}}, w_{\text{sk}}, t_{\text{sk}}$).
- **Soil**: Dry & saturated unit weights ($\gamma_{\text{dry}}, \gamma_{\text{wet}}$), friction angle $\phi$, cohesion $c$, uniform or Boussinesq strip surcharge ($q, B, a$), backfill/toe soil heights, front/back water tables ($H_{\text{wtr}}, H_{\text{wtr,front}}$), and Winkler subgrade modulus parameters ($E_s, \nu$).
- **Structure**: Concrete unit weight $\gamma_c$ and cohesion $c_{\text{concrete}}$.
- **Seismic**: Peak Ground Acceleration ($\text{PGA}$) and site factor ($\text{FPGA}$) for pseudo-static inertial loading ($k_h = \text{PGA} \times \text{FPGA}$).
- **Concrete & Reinf**: Concrete ($f'_c, E_c, f_t$) and steel ($f_{sy}, E_s$) material properties; SNI rebar diameters, spacings, and cover for both **Stem** and **Footing** sections with live bar area calculation.
- **Bearing Capacity**: Optional shallow foundation bearing capacity check (`Sand (Drained)` / `Clay (Undrained)`) and 1D primary consolidation settlement (`Normally Consolidated (NC)` / `Overconsolidated (OC)` with $H_0, e_0, C_c, C_r, p'_c$).
- **Pile**: Optional 2-row (Toe & Heel) pile foundation analysis (`Circle` / `Rectangle`, `Concrete` / `Steel`), pile reinforcement, soil profile, axial/lateral safety factors, and `openpile` static/cyclic loading.

### 2. Right Panel — Dynamic Tabbed Results Workspace
Before running analysis, all calculation tabs remain hidden and only the **ℹ️ Ready** prompt tab (`"Please set parameter and run the analysis"`) is displayed. After clicking **🚀 Run Analysis**, a modal progress splash screen tracks execution and reveals only the active analysis tabs:

1. **📊 Resumed Parameters & KPIs**: Consolidated dashboard of Global Overturning FS, Global Sliding FS, Stem Junction Overturning FS, Bearing Capacity FS (when enabled), Max FEM Nodal Displacements ($X/Y$), Concrete Strength status, FEM Differential Settlement, Footing Rotation, Consolidation Settlement (when enabled), and Max Pile Settlement (when enabled).
2. **📐 Model Geometry & FEM Mesh**: Auto-cropped `drawsvg` dimensioned cross-section with soil/water blocks and centroids, plus a collapsible `opsvis` OpenSees 2D `Tri31` mesh & boundary condition plot.
3. **📉 Stem Bending Moment Diagram**: Internal bending moment distribution along the stem height.
4. **📏 SNI Geometry Verification**: SNI Perencanaan Geoteknik (Gambar 35b) proportion diagram, verification table, and status banner.
5. **🎭 FE Stress Contours**: Side-by-side 2D finite element stress contours ($\sigma_{xx}$ and $\sigma_{yy}$).
6. **🛡️ Stability Verification Report**: Detailed global overturning, global sliding, stem overturning, and bearing capacity checklist with collapsible LaTeX/HTML equilibrium equations.
7. **🌊 Applied Loads & Pressures**: Interactive load-type selector (`total`, `soil lateral`, `soil vertical`, `hydrostatic`, `uplift`, `seismic`), peak loading KPIs, collapsible nodal force tables, and step-by-step Rankine/hydrostatic derivations.
8. **📏 FEM Settlement & Displacement**: Toe/Heel settlement summary, differential settlement check ($\le 50\text{ mm}$ per SNI 8460:2017), rotation tilt, and bottom-node displacement table.
9. **🛡️ Concrete Strength Verification**:
   - Ultimate bending capacity comparison table ($M^*$ vs. $\phi M_u$) for Stem Wall (Base), Footing Toe (Sagging), and Footing Heel (Hogging).
   - **Applied Loads Table**: Breakdown of lateral loads on the Stem (`Fx_soil+q`, `Fx_water`, `kh·W`, FEM base section integration) and vertical loads on the Footing Toe & Heel ($q(x)$ bearing pressure vs. $w_{\text{down}}$ self-weight/backfill/surcharge).
   - **Actual Bending Moment ($M^*$) Calculation**: Collapsible step-by-step formulas and numeric substitutions for $M^*_{\text{stem}}$, base bearing eccentricity $e$, $M^*_{\text{toe}}$, and $M^*_{\text{heel}}$.
   - **Retaining Wall Section-Cut Reference & Rebar Cross-Sections**: Reference wall section-cut drawing (`wall_section_cuts.png`) followed by `concreteproperties` Stem and Footing rebar cross-section plots.
10. **🧱 Pile Foundation Analysis & Design** *(shown only when Pile Analysis is enabled)*: Toe & Heel pile axial/shear/moment KPIs, SVG pile foundation elevation with reaction arrows, 4-panel Winkler response profiles (Deflection, Shear $V$, Moment $M$, Axial Settlement), and Moment-Axial ($M$-$N$) interaction diagram.
11. **📋 Pile Section & Verification Report** *(shown only when Pile Analysis is enabled)*: Merged tab containing the Pile Cross-Section Geometry plot, reinforcement summary, and the 8-point Pile Foundation Verification Report table.
12. **⛰️ Global Stability (2D SSRM)**: XSLOPE slope geometry & inputs plot, FEM mesh summary (`tri6` elements, material zones, pore pressures), material table, and 2D Viscoplastic Shear Strain failure surface contour plot.

### 3. Engineering Report Export (**DOCX** & **PDF**)
Clicking **📄 Generate Report** exports all high-resolution figures and equation renders into `report_temp/`, builds the complete Word report (`retaining_wall_report.docx`), and renders an identical **PDF** report (`retaining_wall_report.pdf`) directly from the generated DOCX document structure using native Qt `QTextDocument` + `QPdfWriter`.

---

## 🛠️ Installation & Running

### Prerequisites
- **Python 3.10 or above**
- Cairo shared library (for `drawsvg` PNG rasterization)
- Required Python packages (`pip install -r requirements.txt`)

### Run the Desktop Application
```bash
python main.py
```

---

## 📦 Open-Source Repositories & Libraries Used

| Library / Package | Repository / Source | Role in QRTWALL-Cantilever |
|---|---|---|
| **PySide6 (Qt for Python)** | https://code.qt.io/cgit/pyside/pyside-setup.git | Native Qt6 desktop GUI, `.ui` dynamic loader (`QUiLoader`), modal progress dialog, and native PDF rendering (`QPdfWriter` / `QTextDocument`). |
| **OpenSeesPy** | https://github.com/zhuminjie/OpenSeesPy (`OpenSees`: https://github.com/OpenSees/OpenSees) | 2D finite element solver (`Tri31` continuum elements, `zeroLength` Winkler subgrade springs, static & pseudo-static seismic analysis). |
| **opsvis** | https://github.com/erduran/opsvis | OpenSees 2D finite element mesh, node/element tag, and stress contour visualization ($\sigma_{xx}, \sigma_{yy}$). |
| **concreteproperties** | https://github.com/robbievanleeuwen/concrete-properties | Reinforced concrete cross-section modeling, ultimate bending capacity ($\phi M_u$), and Moment-Axial ($M$-$N$) interaction envelopes for stem, footing, and piles. |
| **sectionproperties** | https://github.com/robbievanleeuwen/section-properties | Geometric cross-section primitive generation and finite element section mesh routines. |
| **groundhog** | https://github.com/snakesonabrain/groundhog | Geotechnical shallow foundation drained/undrained bearing capacity (`verticalcapacity_drained_api`, `verticalcapacity_undrained_api`) and consolidation settlement theory. |
| **openpile** | https://github.com/ahaapala/openpile | Beam-on-nonlinear-Winkler-foundation ($p$-$y$, $t$-$z$, $Q$-$z$) lateral and axial pile response analysis. |
| **xslope** | https://github.com/normjones/xslope | 2D Finite Element Strength Reduction Method (SSRM) global slope stability solver and viscoplastic shear strain failure surface plotting. |
| **gmsh** | https://gitlab.onelab.info/gmsh/gmsh | 2D unstructured 6-node triangular (`tri6`) finite element mesh generation for global slope stability. |
| **drawsvg** | https://github.com/cduck/drawsvg | Programmatic SVG vector drafting for retaining wall geometry, SNI dimension checks, pressure diagrams, and pile elevation schematics. |
| **CairoSVG / Cairo** | https://github.com/Kozea/CairoSVG | High-DPI SVG-to-PNG rasterization backend used by `drawsvg`. |
| **python-docx** | https://github.com/python-openxml/python-docx | Automated Microsoft Word (`.docx`) engineering report generation. |
| **NumPy / SciPy / pandas / Matplotlib** | https://github.com/numpy/numpy, https://github.com/scipy/scipy, https://github.com/pandas-dev/pandas, https://github.com/matplotlib/matplotlib | Matrix computation, interpolation, tabular data structures, and scientific figure rendering. |


