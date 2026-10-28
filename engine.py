# ----------------------------------------
# Cantilever Retaining Wall
# ----------------------------------------

# Ensure conda Library/bin is in PATH for CairoSVG (and other libraries) on Windows
import os
import sys
conda_prefix = getattr(sys, 'real_prefix', getattr(sys, 'base_prefix', sys.prefix))
lib_bin = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.exists(lib_bin) and lib_bin not in os.environ['PATH']:
    os.environ['PATH'] = lib_bin + os.path.pathsep + os.environ['PATH']

# Importing necessary libraries
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np
if 'st' not in globals():
    import streamlit as st
import drawsvg as draw
import pandas as pd

# Local implementations of Boussinesq strip load stress distribution functions (ensures Groundhog version compatibility)
def stresses_stripload_local(z, x, width, imposedstress, triangular=False):
    z = max(1e-5, float(z))
    R_1 = np.sqrt(x ** 2 + z ** 2)
    R_2 = np.sqrt((x - width) ** 2 + z ** 2)

    _theta1 = np.arccos(z / R_1)
    _theta2 = np.arccos(z / R_2)
    if x < width:
        _theta2 = -_theta2
    beta = _theta2
    alpha = _theta1 - beta

    if triangular:
        _delta_sigma_z = (imposedstress / np.pi) * (
            (x / width) * alpha -
            0.5 * np.sin(2 * beta)
        )
        _delta_sigma_x = (imposedstress / np.pi) * (
            (x / width) * alpha -
            (z / width) * np.log((R_1 ** 2) / (R_2 ** 2)) +
            0.5 * np.sin(2 * beta)
        )
        _delta_tau_zx = (imposedstress / (2 * np.pi)) * (
            1 +
            np.cos(2 * beta) -
            2 * (z / width) * alpha
        )
    else:
        _delta_sigma_z = (imposedstress / np.pi) * (
            alpha +
            np.sin(alpha) * np.cos(alpha + 2 * beta)
        )
        _delta_sigma_x = (imposedstress / np.pi) * (
            alpha -
            np.sin(alpha) * np.cos(alpha + 2 * beta)
        )
        _delta_tau_zx = (imposedstress / np.pi) * (
            np.sin(alpha) * np.sin(alpha + 2 * beta)
        )

    return {
        'delta sigma z [kPa]': _delta_sigma_z,
        'delta sigma x [kPa]': _delta_sigma_x,
        'delta tau zx [kPa]': _delta_tau_zx,
    }

def stresses_stripload_retainingwall_local(imposedstress, width, offset, toe_depth, depth):
    H0 = max(1e-5, float(toe_depth))
    z = max(1e-5, float(depth))
    a = float(offset)
    B = float(width)
    
    alpha_rad = np.arctan(a / z)
    alpha_far_rad = np.arctan((a + B) / z)
    beta_rad = alpha_far_rad - alpha_rad
    alpha_bisector_rad = alpha_rad + beta_rad / 2.0
    
    _delta_sigma_x = (2.0 * imposedstress / np.pi) * (
        beta_rad - np.sin(beta_rad) * np.cos(2.0 * alpha_bisector_rad)
    )
    
    theta_1 = np.degrees(np.arctan(a / H0))
    theta_2 = np.degrees(np.arctan((a + B) / H0))
    R_1 = ((a + B) ** 2) * (90.0 - theta_2)
    R_2 = (a ** 2) * (90.0 - theta_1)
    
    _delta_P_x = (imposedstress / 90.0) * (H0 * (theta_2 - theta_1))
    
    denominator = 2.0 * H0 * (theta_2 - theta_1)
    if abs(denominator) < 1e-5:
        _z_bar = H0 / 2.0
    else:
        _z_bar = (H0**2 * (theta_2 - theta_1) - (R_1 - R_2) + 57.3 * B * H0) / denominator
        
    return {
        'delta sigma x [kPa]': _delta_sigma_x,
        'delta P x [kN/m]': _delta_P_x,
        'z bar [m]': _z_bar,
        'theta 1 [deg]': theta_1,
        'theta 2 [deg]': theta_2,
        'R 1 [m]': R_1,
        'R 2 [m]': R_2,
        'alpha [deg]': np.degrees(alpha_bisector_rad),
        'beta [deg]': np.degrees(beta_rad)
    }

def generate_load_drawing(selected_load_plot):
    mult = 1000
    y_shift = 0.8 * mult
    x_shift = 2.5 * mult
    max_height = max(Hw, h_soil, h_soil_toe, Hwtr, Hwtr_front) + h_ftg
    
    # Calculate global max pressure to plot all load types proportionally
    h_active = min(Hw, h_soil)
    if surcharge_type == 'Strip Load':
        q_top = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - h_active))['delta sigma x [kPa]'] if h_soil > 0 else 0.0
        q_bot = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
    else:
        q_top = Ka * q if h_soil > 0 else 0.0
        q_bot = Ka * q if h_soil > 0 else 0.0
    p_top_act = Ka * (gamma_soil_dry * max(0.0, h_soil - h_active)) + q_top if h_soil > 0 else 0.0
    p_bot_act = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot if h_soil > 0 else 0.0

    FS_passive_val = 2.0
    Kp_val = np.tan(np.radians(45.0 + phi / 2.0)) ** 2
    p_p_bot_act = (Kp_val * gamma_soil_dry * (h_soil_toe + h_ftg)) / FS_passive_val

    p_v_heel = q_soil + q
    p_w_back_act = gamma_w * Hwtr
    p_w_front_act = gamma_w * Hwtr_front
    p_upl_act = gamma_w * Hwtr

    P_max_global = max(p_top_act, p_bot_act, p_p_bot_act, p_v_heel, p_w_back_act, p_w_front_act, p_upl_act, 30.0)
    scale_press = 1.0 * mult / P_max_global

    # Initialize drawing
    d_load = draw.Drawing((ftg + 4.5)*mult, (max_height + 3.0)*mult, origin='bottom-left')
    
    # Helper for arrows
    def draw_arrow_head(d, x1, y1, x2, y2, color='red', stroke_width=15, head_len=80, head_width=50):
        d.append(draw.Line(x1, y1, x2, y2, stroke=color, stroke_width=stroke_width))
        # Calculate arrowhead
        dx = x2 - x1
        dy = y2 - y1
        L = np.hypot(dx, dy)
        if L < 1e-6:
            return
        ux = dx / L
        uy = dy / L
        # Perp vector
        px = -uy
        py = ux
        # Arrow head points
        ax = x2 - head_len * ux
        ay = y2 - head_len * uy
        
        p1x = ax + head_width * px
        p1y = ay + head_width * py
        p2x = ax - head_width * px
        p2y = ay - head_width * py
        
        d.append(draw.Lines(x2, y2, p1x, p1y, p2x, p2y, close=True, fill=color, stroke=color))

    # 1. Background Soil outlines/shapes (very light)
    # Soil dry
    if 'soil_dry_poly' in globals() and soil_dry_poly:
        pts_dry = []
        for px, py in soil_dry_poly:
            pts_dry.extend([px*mult + x_shift, py*mult - y_shift])
        d_load.append(draw.Lines(*pts_dry, close=True, fill='#5BC2A5', fill_opacity=0.06, stroke='#bdc3c7', stroke_width=2, stroke_dasharray='10,10'))
    # Soil wet
    if 'soil_wet_poly' in globals() and soil_wet_poly:
        pts_wet = []
        for px, py in soil_wet_poly:
            pts_wet.extend([px*mult + x_shift, py*mult - y_shift])
        d_load.append(draw.Lines(*pts_wet, close=True, fill='#A6F527', fill_opacity=0.06, stroke='#bdc3c7', stroke_width=2, stroke_dasharray='10,10'))
    # Soil toe
    if h_soil_toe > 0:
        d_load.append(draw.Lines(x_shift, -h_ftg*mult - y_shift,
                             toe*mult + x_shift, -h_ftg*mult - y_shift,
                             toe*mult + x_shift, -(h_ftg + h_soil_toe)*mult - y_shift,
                             x_shift, -(h_ftg + h_soil_toe)*mult - y_shift,
                             close=True, fill='#5BC2A5', fill_opacity=0.06, stroke='#bdc3c7', stroke_width=2, stroke_dasharray='10,10'))

    # Water level lines (dashed)
    if Hwtr_front > 0:
        d_load.append(draw.Line(x_shift, -(h_ftg + Hwtr_front)*mult - y_shift,
                                toe*mult + x_shift, -(h_ftg + Hwtr_front)*mult - y_shift,
                                stroke='blue', stroke_width=10, stroke_dasharray='30,30'))
    if Hwtr > 0:
        d_load.append(draw.Line((toe+bot_wall)*mult + x_shift, -(h_ftg + Hwtr)*mult - y_shift,
                                ftg*mult + x_shift, -(h_ftg + Hwtr)*mult - y_shift,
                                stroke='blue', stroke_width=10, stroke_dasharray='30,30'))

    # 2. Draw yellow wall
    if taper_direction == 'Toe-facing':
        stem_pts_local = [
            ((toe + taper/4.0), -(h_ftg + Hw/4.0)),
            ((toe + taper/2.0), -(h_ftg + Hw/2.0)),
            ((toe + taper*3.0/4.0), -(h_ftg + Hw*3.0/4.0)),
            ((toe + taper), -(h_ftg + Hw)),
            ((toe + bot_wall), -(h_ftg + Hw)),
            ((toe + bot_wall), -(h_ftg + Hw*3.0/4.0)),
            ((toe + bot_wall), -(h_ftg + Hw/2.0)),
            ((toe + bot_wall), -(h_ftg + Hw/4.0)),
            ((toe + bot_wall), -h_ftg)
        ]
    else:
        stem_pts_local = [
            (toe, -(h_ftg + Hw/4.0)),
            (toe, -(h_ftg + Hw/2.0)),
            (toe, -(h_ftg + Hw*3.0/4.0)),
            (toe, -(h_ftg + Hw)),
            ((toe + top_wall), -(h_ftg + Hw)),
            ((toe + bot_wall - taper*3.0/4.0), -(h_ftg + Hw*3.0/4.0)),
            ((toe + bot_wall - taper/2.0), -(h_ftg + Hw/2.0)),
            ((toe + bot_wall - taper/4.0), -(h_ftg + Hw/4.0)),
            ((toe + bot_wall), -h_ftg)
        ]
    
    wall_points_load = [
        x_shift,  -y_shift,
        x_shift,  -h_ftg*mult - y_shift,
        toe*mult + x_shift, -h_ftg*mult - y_shift
    ]
    for px, py in stem_pts_local:
        wall_points_load.extend([px*mult + x_shift, py*mult - y_shift])
    wall_points_load.extend([
        ftg*mult + x_shift, -h_ftg*mult - y_shift,
        ftg*mult + x_shift, -y_shift
    ])
    
    d_load.append(draw.Lines(*wall_points_load,
                        close=True,
                        fill='#eeee00',
                        stroke='black',
                        stroke_width=30))

    # Draw Shear Key in d_load if enabled
    if include_shear_key:
        sk_x1 = shear_key_distance * mult + x_shift
        sk_x2 = (shear_key_distance + shear_key_width) * mult + x_shift
        sk_y1 = -y_shift
        sk_y2 = -y_shift + shear_key_thickness * mult
        d_load.append(draw.Lines(sk_x1, sk_y1,
                            sk_x2, sk_y1,
                            sk_x2, sk_y2,
                            sk_x1, sk_y2,
                            close=True,
                            fill='#FFD700',
                            stroke='black',
                            stroke_width=20))

    # 3. Dynamic loads
    if selected_load_plot in ['soil lateral', 'total']:
        h_active = min(Hw, h_soil)
        if surcharge_type == 'Strip Load':
            q_top = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - h_active))['delta sigma x [kPa]'] if h_soil > 0 else 0.0
            q_bot = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
        else:
            q_top = Ka * q if h_soil > 0 else 0.0
            q_bot = Ka * q if h_soil > 0 else 0.0
        
        p_top = Ka * (gamma_soil_dry * max(0.0, h_soil - h_active)) + q_top if h_soil > 0 else 0.0
        p_bot = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot if h_soil > 0 else 0.0
        
        FS_passive_val = 2.0
        Kp_val = np.tan(np.radians(45.0 + phi / 2.0)) ** 2
        p_p_top = (Kp_val * gamma_soil_dry * h_soil_toe) / FS_passive_val if h_soil_toe > 0 else 0.0
        p_p_bot = (Kp_val * gamma_soil_dry * (h_soil_toe + h_ftg)) / FS_passive_val
        
        p_max_lat = max(p_top, p_bot, p_p_bot, 1.0)
        scale_lat = scale_press
        
        d_load.append(draw.Lines((toe+bot_wall)*mult + x_shift, -(h_ftg+h_active)*mult - y_shift,
                            (toe+bot_wall)*mult + x_shift + p_top*scale_lat, -(h_ftg+h_active)*mult - y_shift,
                            (toe+bot_wall)*mult + x_shift + p_bot*scale_lat, -h_ftg*mult - y_shift,
                            (toe+bot_wall)*mult + x_shift, -h_ftg*mult - y_shift,
                            close=True, fill='#E67E22', fill_opacity=0.3, stroke='#D35400', stroke_width=12))
        
        n_arr = 5
        for i in range(n_arr):
            frac = i / (n_arr - 1)
            z_val = frac * h_active
            if h_soil > 0:
                hd = max(0.0, h_soil - max(z_val, Hwtr))
                hw = max(0.0, min(h_soil, Hwtr) - z_val)
                if surcharge_type == 'Strip Load':
                    q_z = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - z_val))['delta sigma x [kPa]']
                else:
                    q_z = Ka * q
                pz = Ka * (gamma_soil_dry * hd + gamma_soil_wet * hw) + q_z
            else:
                pz = 0.0
            y_z = -(h_ftg + z_val)*mult - y_shift
            x_start = (toe + bot_wall)*mult + x_shift + pz * scale_lat
            x_end = (toe + bot_wall)*mult + x_shift
            if pz > 0:
                draw_arrow_head(d_load, x_start, y_z, x_end, y_z, color='#D35400', stroke_width=10, head_len=60, head_width=35)
        
        d_load.append(draw.Lines(x_shift, -(h_ftg + h_soil_toe)*mult - y_shift,
                            x_shift - p_p_top*scale_lat, -(h_ftg + h_soil_toe)*mult - y_shift,
                            x_shift - p_p_bot*scale_lat, -h_ftg*mult - y_shift,
                            x_shift, -h_ftg*mult - y_shift,
                            close=True, fill='#2ECC71', fill_opacity=0.3, stroke='#27AE60', stroke_width=12))
        
        n_arr_p = 3
        for i in range(n_arr_p):
            frac = i / (n_arr_p - 1)
            z_val = frac * (h_soil_toe + h_ftg)
            depth_p = (h_soil_toe + h_ftg) - z_val
            pz_p = (Kp_val * gamma_soil_dry * depth_p) / FS_passive_val
            y_z = -z_val*mult - y_shift
            x_start = x_shift - pz_p * scale_lat
            x_end = x_shift
            if pz_p > 0:
                draw_arrow_head(d_load, x_start, y_z, x_end, y_z, color='#27AE60', stroke_width=10, head_len=60, head_width=35)

        d_load.append(draw.Text(f"{p_top:.2f} kPa", 0.20*mult, (toe+bot_wall)*mult + x_shift + p_top*scale_lat + 0.20*mult, -(h_ftg+h_active)*mult - y_shift, fill='#D35400', font_weight='bold'))
        d_load.append(draw.Text(f"{p_bot:.2f} kPa", 0.20*mult, (toe+bot_wall)*mult + x_shift + p_bot*scale_lat + 0.20*mult, -h_ftg*mult - y_shift, fill='#D35400', font_weight='bold'))
        d_load.append(draw.Text(f"{p_p_top:.2f} kPa", 0.20*mult, x_shift - p_p_top*scale_lat - 0.05*mult, -(h_ftg+h_soil_toe)*mult - y_shift, fill='#27AE60', font_weight='bold', text_anchor='end'))
        d_load.append(draw.Text(f"{p_p_bot:.2f} kPa", 0.20*mult, x_shift - p_p_bot*scale_lat - 0.05*mult, -h_ftg*mult - y_shift, fill='#27AE60', font_weight='bold', text_anchor='end'))

    if selected_load_plot in ['soil vertical', 'total']:
        p_v = q_soil + q
        scale_v = scale_press
        d_load.append(draw.Lines((toe+bot_wall)*mult + x_shift, -h_ftg*mult - y_shift,
                            (toe+bot_wall)*mult + x_shift, -h_ftg*mult - y_shift - p_v*scale_v,
                            ftg*mult + x_shift, -h_ftg*mult - y_shift - p_v*scale_v,
                            ftg*mult + x_shift, -h_ftg*mult - y_shift,
                            close=True, fill='#F39C12', fill_opacity=0.3, stroke='#E67E22', stroke_width=12))
        
        n_arr = 5
        for i in range(n_arr):
            frac = i / (n_arr - 1)
            x_pos = (toe + bot_wall + frac * heel)*mult + x_shift
            y_start = -h_ftg*mult - y_shift - p_v*scale_v
            y_end = -h_ftg*mult - y_shift
            draw_arrow_head(d_load, x_pos, y_start, x_pos, y_end, color='#E67E22', stroke_width=10, head_len=60, head_width=35)
        
        d_load.append(draw.Text(f"{p_v:.2f} kPa", 0.20*mult, (toe+bot_wall + heel/2)*mult + x_shift, -h_ftg*mult - y_shift - p_v*scale_v - 0.22*mult, fill='#E67E22', text_anchor='middle', font_weight='bold'))

    if selected_load_plot in ['hydrostatic', 'total']:
        p_w_back = gamma_w * Hwtr
        p_w_front = gamma_w * Hwtr_front
        p_max_w = max(p_w_back, p_w_front, 1.0)
        scale_w = scale_press
        
        if Hwtr > 0:
            d_load.append(draw.Lines((toe+bot_wall)*mult + x_shift, -(h_ftg+Hwtr)*mult - y_shift,
                                (toe+bot_wall)*mult + x_shift + p_w_back*scale_w, -h_ftg*mult - y_shift,
                                (toe+bot_wall)*mult + x_shift, -h_ftg*mult - y_shift,
                                close=True, fill='#3498DB', fill_opacity=0.3, stroke='#2980B9', stroke_width=12))
            n_arr = 3
            for i in range(n_arr):
                frac = i / (n_arr - 1)
                z_val = frac * Hwtr
                pz_w = gamma_w * (Hwtr - z_val)
                y_z = -(h_ftg + z_val)*mult - y_shift
                x_start = (toe + bot_wall)*mult + x_shift + pz_w * scale_w
                x_end = (toe + bot_wall)*mult + x_shift
                if pz_w > 0:
                    draw_arrow_head(d_load, x_start, y_z, x_end, y_z, color='#2980B9', stroke_width=10, head_len=50, head_width=30)
            
            d_load.append(draw.Text(f"{p_w_back:.2f} kPa", 0.20*mult, (toe+bot_wall)*mult + x_shift + p_w_back*scale_w + 0.20*mult, -(h_ftg - 0.25)*mult - y_shift, fill='#2980B9', font_weight='bold'))

        if Hwtr_front > 0:
            d_load.append(draw.Lines(toe*mult + x_shift, -(h_ftg+Hwtr_front)*mult - y_shift,
                                toe*mult + x_shift - p_w_front*scale_w, -h_ftg*mult - y_shift,
                                toe*mult + x_shift, -h_ftg*mult - y_shift,
                                close=True, fill='#3498DB', fill_opacity=0.3, stroke='#2980B9', stroke_width=12))
            n_arr = 3
            for i in range(n_arr):
                frac = i / (n_arr - 1)
                z_val = frac * Hwtr_front
                pz_w = gamma_w * (Hwtr_front - z_val)
                y_z = -(h_ftg + z_val)*mult - y_shift
                x_start = toe*mult + x_shift - pz_w * scale_w
                x_end = toe*mult + x_shift
                if pz_w > 0:
                    draw_arrow_head(d_load, x_start, y_z, x_end, y_z, color='#2980B9', stroke_width=10, head_len=50, head_width=30)
            
            d_load.append(draw.Text(f"{p_w_front:.2f} kPa", 0.20*mult, toe*mult + x_shift - p_w_front*scale_w - 0.05*mult, -h_ftg*mult - y_shift, fill='#2980B9', font_weight='bold', text_anchor='end'))

    if selected_load_plot in ['uplift', 'total']:
        p_upl = gamma_w * Hwtr
        if p_upl > 0:
            scale_u = scale_press
            d_load.append(draw.Lines(x_shift, -y_shift,
                                x_shift, -y_shift + p_upl*scale_u,
                                ftg*mult + x_shift, -y_shift + p_upl*scale_u,
                                ftg*mult + x_shift, -y_shift,
                                close=True, fill='#9B59B6', fill_opacity=0.3, stroke='#8E44AD', stroke_width=12))
            
            n_arr = 6
            for i in range(n_arr):
                frac = i / (n_arr - 1)
                x_pos = frac * ftg * mult + x_shift
                y_start = -y_shift + p_upl*scale_u
                y_end = -y_shift
                draw_arrow_head(d_load, x_pos, y_start, x_pos, y_end, color='#8E44AD', stroke_width=10, head_len=50, head_width=30)
                
            d_load.append(draw.Text(f"{p_upl:.2f} kPa", 0.20*mult, (ftg/2)*mult + x_shift, -y_shift + p_upl*scale_u + 0.25*mult, fill='#8E44AD', text_anchor='middle', font_weight='bold'))

    if selected_load_plot in ['seismic', 'total']:
        if kh > 0:
            if stem_centroid[0] is not None:
                sx, sy = stem_centroid[0]*mult + x_shift, stem_centroid[1]*mult - y_shift
                draw_arrow_head(d_load, sx + 0.8*mult, sy, sx, sy, color='red', stroke_width=15, head_len=80, head_width=45)
                d_load.append(draw.Text("Seismic (stem)", 0.18*mult, sx + 0.95*mult, sy + 0.05*mult, fill='red', font_weight='bold'))
            if base_centroid[0] is not None:
                bx, by = base_centroid[0]*mult + x_shift, base_centroid[1]*mult - y_shift
                draw_arrow_head(d_load, bx + 0.8*mult, by, bx, by, color='red', stroke_width=15, head_len=80, head_width=45)
                d_load.append(draw.Text("Seismic (base)", 0.18*mult, bx + 0.95*mult, by + 0.05*mult, fill='red', font_weight='bold'))
            if 'soil_dry_centroid' in globals() and soil_dry_centroid[0] is not None:
                sdx, sdy = soil_dry_centroid[0]*mult + x_shift, soil_dry_centroid[1]*mult - y_shift
                draw_arrow_head(d_load, sdx + 0.8*mult, sdy, sdx, sdy, color='red', stroke_width=15, head_len=80, head_width=45)
                d_load.append(draw.Text("Seismic (soil dry)", 0.18*mult, sdx + 0.95*mult, sdy + 0.05*mult, fill='red', font_weight='bold'))
            elif 'soil_dry_centroid' in locals() and soil_dry_centroid[0] is not None:
                sdx, sdy = soil_dry_centroid[0]*mult + x_shift, soil_dry_centroid[1]*mult - y_shift
                draw_arrow_head(d_load, sdx + 0.8*mult, sdy, sdx, sdy, color='red', stroke_width=15, head_len=80, head_width=45)
                d_load.append(draw.Text("Seismic (soil dry)", 0.18*mult, sdx + 0.95*mult, sdy + 0.05*mult, fill='red', font_weight='bold'))

            if 'soil_wet_centroid' in globals() and soil_wet_centroid[0] is not None:
                swx, swy = soil_wet_centroid[0]*mult + x_shift, soil_wet_centroid[1]*mult - y_shift
                draw_arrow_head(d_load, swx + 0.8*mult, swy, swx, swy, color='red', stroke_width=15, head_len=80, head_width=45)
                d_load.append(draw.Text("Seismic (soil wet)", 0.18*mult, swx + 0.95*mult, swy + 0.05*mult, fill='red', font_weight='bold'))
            elif 'soil_wet_centroid' in locals() and soil_wet_centroid[0] is not None:
                swx, swy = soil_wet_centroid[0]*mult + x_shift, soil_wet_centroid[1]*mult - y_shift
                draw_arrow_head(d_load, swx + 0.8*mult, swy, swx, swy, color='red', stroke_width=15, head_len=80, head_width=45)
                d_load.append(draw.Text("Seismic (soil wet)", 0.18*mult, swx + 0.95*mult, swy + 0.05*mult, fill='red', font_weight='bold'))

    if selected_load_plot == 'total':
        if base_centroid[0] is not None:
            bx, by = base_centroid[0]*mult + x_shift, base_centroid[1]*mult - y_shift
            draw_arrow_head(d_load, bx, by - 0.8*mult, bx, by, color='blue', stroke_width=15, head_len=80, head_width=45)
            d_load.append(draw.Text(f"W_base = {W_base:.1f} kN", 0.18*mult, bx, by - 1.0*mult, fill='blue', text_anchor='middle'))
        if stem_centroid[0] is not None:
            sx, sy = stem_centroid[0]*mult + x_shift, stem_centroid[1]*mult - y_shift
            draw_arrow_head(d_load, sx, sy - 0.8*mult, sx, sy, color='blue', stroke_width=15, head_len=80, head_width=45)
            d_load.append(draw.Text(f"W_stem = {W_stem_rect:.1f} kN", 0.18*mult, sx, sy - 1.0*mult, fill='blue', text_anchor='middle'))

    return d_load

def generate_latex_equation_images():
    import matplotlib.pyplot as plt
    import os
    os.makedirs("report_temp", exist_ok=True)
    
    def render_eq(latex_str, filename):
        fig, ax = plt.subplots(figsize=(6, 0.9))
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')
        if not latex_str.startswith("$"):
            latex_str = f"${latex_str}$"
        ax.text(0.5, 0.5, latex_str, fontsize=14, ha='center', va='center', color='#162447')
        ax.axis('off')
        path = os.path.join("report_temp", filename)
        fig.savefig(path, dpi=200, bbox_inches='tight', transparent=True)
        plt.close(fig)

    try:
        render_eq(r"K_a = \tan^2\left(45^\circ - \frac{\phi}{2}\right)", "eq_ka.png")
        render_eq(r"p_{act,top} = K_a \cdot q", "eq_p_act_top.png")
        render_eq(r"p_{act,bot} = K_a \cdot (\gamma_{dry} z_{dry} + \gamma_{wet} z_{wet}) + K_a \cdot q", "eq_p_act_bot.png")
        render_eq(r"K_p = \tan^2\left(45^\circ + \frac{\phi}{2}\right)", "eq_kp.png")
        render_eq(r"p_{pass,bot} = \frac{K_p \cdot \gamma_{dry} \cdot (h_{toe} + h_{ftg})}{FS_{pass}}", "eq_p_pass_bot.png")
        render_eq(r"p_{v,heel} = q_{soil} + q", "eq_p_v_heel.png")
        render_eq(r"p_{w} = \gamma_w \cdot H_{wtr}", "eq_p_w.png")
        render_eq(r"p_{uplift} = \gamma_w \cdot H_{wtr}", "eq_p_uplift.png")
        render_eq(r"FS_{sliding} = \frac{\sum R_{resisting}}{\sum F_{driving}} = \frac{R_{sliding} + P_{passive}}{F_{driving}}", "eq_sliding.png")
        render_eq(r"FS_{overturning} = \frac{\sum M_{resisting}}{\sum M_{overturning}} = \frac{M_{res}}{M_{ot}}", "eq_overturning.png")
        render_eq(r"\sigma_{max,min} = \frac{R_v}{B} \left( 1 \pm \frac{6e}{B} \right)", "eq_bearing_stress.png")
        render_eq(r"FS_{bearing} = \frac{q_{ult}}{\sigma_{max}}", "eq_bearing_fs.png")
        render_eq(r"S_c = \frac{C_c \cdot H_c}{1 + e_0} \log\left( \frac{\sigma'_{v0} + \Delta \sigma}{\sigma'_{v0}} \right)", "eq_settlement.png")
    except Exception as e_eq:
        print(f"Error rendering latex equations: {e_eq}")

def export_report_assets():
    import os
    import cairosvg
    os.makedirs("report_temp", exist_ok=True)
    
    # Save geometry drawings
    if 'd' in globals() and d is not None:
        try:
            d.set_render_size(800, 550)
            cairosvg.svg2png(bytestring=d.as_svg(), write_to="report_temp/model_geometry.png", output_width=800)
        except Exception:
            pass
    if 'd_sni' in globals() and d_sni is not None:
        try:
            d_sni.set_render_size(800, 550)
            cairosvg.svg2png(bytestring=d_sni.as_svg(), write_to="report_temp/typical_dimensions.png", output_width=800)
        except Exception:
            pass
            
    # Save load drawings (scaled down for MS Word display compatibility)
    for lt in ["total", "soil lateral", "soil vertical", "hydrostatic", "uplift", "seismic"]:
        try:
            d_l = generate_load_drawing(lt)
            if d_l is not None:
                d_l.set_render_size(800, 550)
                filename = f"load_{lt.replace(' ', '_')}.png"
                cairosvg.svg2png(bytestring=d_l.as_svg(), write_to=os.path.join("report_temp", filename), output_width=800)
        except Exception:
            pass
            
    # Save pile drawings
    if 'd_pile_svg' in globals() and d_pile_svg is not None:
        try:
            d_pile_svg.set_render_size(800, 550)
            cairosvg.svg2png(bytestring=d_pile_svg.as_svg(), write_to="report_temp/pile_elevation.png", output_width=800)
        except Exception:
            pass
            
    # Save matplotlib figures (make sure they are updated)
    if 'fig_inputs' in globals() and fig_inputs is not None:
        try:
            fig_inputs.savefig("report_temp/xslope_inputs.png", dpi=150, bbox_inches='tight')
        except Exception:
            pass
    if 'fig_mesh' in globals() and fig_mesh is not None:
        try:
            fig_mesh.savefig("report_temp/xslope_mesh.png", dpi=150, bbox_inches='tight')
        except Exception:
            pass
    if 'fig_winkler' in globals() and fig_winkler is not None:
        try:
            fig_winkler.savefig("report_temp/pile_winkler.png", dpi=150, bbox_inches='tight')
        except Exception:
            pass
    if 'fig_ssrm' in globals() and fig_ssrm is not None:
        try:
            fig_ssrm.savefig("report_temp/stability_ssrm.png", dpi=150, bbox_inches='tight')
        except Exception:
            pass
    if 'fig_pile_sec' in globals() and fig_pile_sec is not None:
        try:
            fig_pile_sec.savefig("report_temp/pile_section.png", dpi=150, bbox_inches='tight')
        except Exception:
            pass

    generate_latex_equation_images()

def build_latex_code():
    import re
    def esc(text):
        if not isinstance(text, str):
            text = str(text)
        conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
        return "".join(conv.get(c, c) for c in text)

    tex = []
    tex.append(r"\documentclass[10pt,a4paper]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{amsmath}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{float}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{fancyhdr}")
    tex.append(r"\usepackage{caption}")
    
    tex.append(r"\pagestyle{fancy}")
    tex.append(r"\fancyhf{}")
    tex.append(r"\lhead{Cantilever Retaining Wall Engineering Report}")
    tex.append(r"\rhead{\thepage}")
    
    tex.append(r"\begin{document}")
    
    # Title Page
    tex.append(r"\title{\textbf{Retaining Wall Design \& Verification Report}}")
    tex.append(r"\author{RT Wall Cantilever}")
    tex.append(r"\date{\today}")
    tex.append(r"\maketitle")
    tex.append(r"\tableofcontents")
    tex.append(r"\newpage")
    
    # Section 1: Input Parameters
    tex.append(r"\section{Input Parameters}")
    tex.append(r"The following design parameters were specified for this analysis:")
    
    # Geometry Table
    tex.append(r"\subsection{Wall Geometry}")
    tex.append(r"\begin{longtable}{lll}")
    tex.append(r"\toprule")
    tex.append(r"Parameter & Symbol & Value \\")
    tex.append(r"\midrule")
    tex.append(f"Out-of-plane thickness & $t$ & {t:.3f} m \\\\")
    tex.append(f"Wall height & $H_w$ & {Hw:.3f} m \\\\")
    tex.append(f"Toe length & $L_{{toe}}$ & {toe:.3f} m \\\\")
    tex.append(f"Heel length & $L_{{heel}}$ & {heel:.3f} m \\\\")
    tex.append(f"Top wall thickness & $b_{{top}}$ & {top_wall:.3f} m \\\\")
    tex.append(f"Bottom wall thickness & $b_{{bot}}$ & {bot_wall:.3f} m \\\\")
    tex.append(f"Footing thickness & $h_{{ftg}}$ & {h_ftg:.3f} m \\\\")
    if include_shear_key:
        tex.append(f"Shear key distance & $x_{{sk}}$ & {shear_key_distance:.3f} m \\\\")
        tex.append(f"Shear key width & $w_{{sk}}$ & {shear_key_width:.3f} m \\\\")
        tex.append(f"Shear key thickness & $t_{{sk}}$ & {shear_key_thickness:.3f} m \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{longtable}")

    # SNI Typical Geometry verification checklist table
    H_tot = Hw + h_ftg
    tex.append(r"\subsection{SNI Typical Geometry Verification Checklist}")
    tex.append(r"Indonesian Geotechnical Standard (SNI Perencanaan Geoteknik) dimension ratios verification checklist:")
    tex.append(r"\begin{longtable}{lllll}")
    tex.append(r"\toprule")
    tex.append(r"Parameter Name & Symbol & Input Value & SNI Requirement & Status \\")
    tex.append(r"\midrule")
    tex.append(f"Top Wall Thickness & $b_{{top}}$ & {top_wall:.2f} m & $\\ge 0.30$ m & {{'PASS' if top_wall >= 0.30 else 'FAIL'}} \\\\")
    tex.append(f"Base Stem Thickness & $b_{{bot}}$ & {bot_wall:.2f} m & $\\ge 0.1 H$ ({0.1*H_tot:.2f} m) & {{'PASS' if bot_wall >= 0.1*H_tot else 'FAIL'}} \\\\")
    slope_val = taper / Hw if (Hw > 0 and taper_direction == 'Toe-facing') else 0.0
    slope_status = 'PASS' if (taper_direction == 'Heel-facing' or slope_val >= (1.0/48.0)) else 'FAIL'
    slope_req = r"\ge 1:48\text{ (0.0208)}" if taper_direction == 'Toe-facing' else "N/A (Vertical)"
    tex.append(f"Front Face Batter Slope & $\\text{{slope}}$ & {slope_val:.4f} & ${slope_req}$ & {slope_status} \\\\")
    tex.append(f"Footing Width & $B$ & {ftg:.2f} m & $0.4 H \\sim 0.7 H$ ({0.4*H_tot:.2f} $\\sim$ {0.7*H_tot:.2f} m) & {{'PASS' if 0.4*H_tot <= ftg <= 0.7*H_tot else 'FAIL'}} \\\\")
    tex.append(f"Footing Thickness & $h_{{ftg}}$ & {h_ftg:.2f} m & $H/12 \\sim H/10$ ({H_tot/12:.2f} $\\sim$ {H_tot/10:.2f} m) & {{'PASS' if H_tot/12 <= h_ftg <= H_tot/10 else 'FAIL'}} \\\\")
    tex.append(f"Toe Slab Length & $L_{{toe}}$ & {toe:.2f} m & $\\ge B/3$ ({ftg/3:.2f} m) & {{'PASS' if toe >= ftg/3 else 'FAIL'}} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{longtable}")
    
    # Soil Table
    tex.append(r"\subsection{Soil \& Water Parameters}")
    tex.append(r"\begin{longtable}{lll}")
    tex.append(r"\toprule")
    tex.append(r"Parameter & Symbol & Value \\")
    tex.append(r"\midrule")
    tex.append(f"Dry unit weight & $\\gamma_{{dry}}$ & {gamma_soil_dry:.2f} kN/m$^3$ \\\\")
    tex.append(f"Wet unit weight & $\\gamma_{{wet}}$ & {gamma_soil_wet:.2f} kN/m$^3$ \\\\")
    tex.append(f"Friction angle & $\\phi$ & {phi:.1f}$^\circ$ \\\\")
    tex.append(f"Soil cohesion & $c$ & {c_soil:.1f} kPa \\\\")
    tex.append(f"Surcharge load & $q$ & {q:.2f} kPa \\\\")
    if surcharge_type == 'Strip Load':
        tex.append(f"Surcharge width & $B_q$ & {width_surcharge:.2f} m \\\\")
        tex.append(f"Surcharge offset & $a_q$ & {offset_surcharge:.2f} m \\\\")
    tex.append(f"Soil height above heel & $h_{{soil}}$ & {h_soil:.3f} m \\\\")
    tex.append(f"Soil above toe & $h_{{toe}}$ & {h_soil_toe:.3f} m \\\\")
    tex.append(f"Water unit weight & $\\gamma_w$ & {gamma_w:.2f} kN/m$^3$ \\\\")
    tex.append(f"Water height (heel) & $H_{{wtr}}$ & {Hwtr:.2f} m \\\\")
    tex.append(f"Water height (toe) & $H_{{wtr,front}}$ & {Hwtr_front:.3f} m \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{longtable}")

    # Concrete and Reinf
    tex.append(r"\subsection{Concrete \& Reinforcement Parameters}")
    tex.append(r"\begin{longtable}{lll}")
    tex.append(r"\toprule")
    tex.append(r"Parameter & Symbol & Value \\")
    tex.append(r"\midrule")
    tex.append(f"Concrete strength & $f'_c$ & {fc:.1f} MPa \\\\")
    tex.append(f"Elastic modulus (concrete) & $E_c$ & {Ec/1000:.1f} GPa \\\\")
    tex.append(f"Steel yield strength & $f_y$ & {fy:.1f} MPa \\\\")
    tex.append(f"Stem main rebar dia & $d_{{stem}}$ & {stem_rebar_dia} mm \\\\")
    tex.append(f"Stem bar spacing & $s_{{stem}}$ & {stem_spacing_x} mm \\\\")
    tex.append(f"Footing main rebar dia & $d_{{ftg}}$ & {ftg_rebar_dia} mm \\\\")
    tex.append(f"Footing bar spacing & $s_{{ftg}}$ & {ftg_spacing_x} mm \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{longtable}")
    
    # Section 2: Model and geometries image
    tex.append(r"\newpage")
    tex.append(r"\section{Model \& Geometry Diagrams}")
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.8\textwidth]{report_temp/model_geometry.png}")
    tex.append(r"\caption{Retaining Wall Geometry and Mesh Outline}")
    tex.append(r"\end{figure}")
    
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.8\textwidth]{report_temp/typical_dimensions.png}")
    tex.append(r"\caption{Dimensions Verification per SNI Geotechnical Standards}")
    tex.append(r"\end{figure}")
    
    # Section 3: Applied Loads & Pressure Distributions
    tex.append(r"\newpage")
    tex.append(r"\section{Applied Loads \& Pressure Distributions}")
    tex.append(r"The boundary loads and pressure profiles are divided by loading type:")

    load_types = [
        ("soil_lateral", "Soil Lateral Pressure (Active/Passive)"),
        ("soil_vertical", "Soil Vertical Load"),
        ("hydrostatic", "Hydrostatic Water Pressure"),
        ("uplift", "Uplift Pressure"),
        ("seismic", "Seismic Force"),
        ("total", "Total / Combined Loading")
    ]
    for key, name in load_types:
        tex.append(f"\\subsection{{{name}}}")
        tex.append(f"The load diagram for {name} is shown below:")
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(f"\\includegraphics[width=0.7\\textwidth]{{report_temp/load_{key}.png}}")
        tex.append(f"\\caption{{{name} Diagram}}")
        tex.append(r"\end{figure}")

    # Analytical load calculation breakdowns
    tex.append(r"\subsection{Analytical Load Calculation Breakdown}")
    
    # 1. Soil Lateral Pressure Calculation
    tex.append(r"\subsubsection{Active \& Passive Lateral Earth Pressure}")
    tex.append(r"Active and Passive lateral earth pressure parameters calculated per Terzaghi / Rankine equations:")
    tex.append(r"\begin{itemize}")
    tex.append(f"\\item Coefficient of active earth pressure: $K_a = \\tan^2(45^\\circ - \\phi/2) = {Ka:.4f}$")
    h_active = min(Hw, h_soil)
    if surcharge_type == 'Strip Load':
        q_top = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - h_active))['delta sigma x [kPa]'] if h_soil > 0 else 0.0
        q_bot = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
    else:
        q_top = Ka * q if h_soil > 0 else 0.0
        q_bot = Ka * q if h_soil > 0 else 0.0
    p_top_act = Ka * (gamma_soil_dry * max(0.0, h_soil - h_active)) + q_top if h_soil > 0 else 0.0
    p_bot_act = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot if h_soil > 0 else 0.0
    tex.append(f"\\item Active lateral pressure at stem top ($p_{{act,top}}$): $K_a \\cdot (\\gamma \\cdot z_{{top}}) + K_a \\cdot q = {p_top_act:.2f}$ kPa")
    tex.append(f"\\item Active lateral pressure at footing bottom ($p_{{act,bot}}$): $K_a \\cdot (\\gamma \\cdot z_{{bot}}) + K_a \\cdot q = {p_bot_act:.2f}$ kPa")
    
    FS_passive_val = 2.0
    Kp_val = np.tan(np.radians(45.0 + phi / 2.0)) ** 2
    p_p_top = (Kp_val * gamma_soil_dry * h_soil_toe) / FS_passive_val if h_soil_toe > 0 else 0.0
    p_p_bot = (Kp_val * gamma_soil_dry * (h_soil_toe + h_ftg)) / FS_passive_val
    tex.append(f"\\item Coefficient of passive earth pressure: $K_p = \\tan^2(45^\\circ + \\phi/2) = {Kp_val:.4f}$")
    tex.append(f"\\item Mobilized passive lateral pressure at toe top ($p_{{pass,top}}$): $K_p \\cdot \\gamma_{{dry}} \\cdot h_{{toe}} / FS_{{pass}} = {p_p_top:.2f}$ kPa")
    tex.append(f"\\item Mobilized passive lateral pressure at footing bottom ($p_{{pass,bot}}$): $K_p \\cdot \\gamma_{{dry}} \\cdot (h_{{toe}} + h_{{ftg}}) / FS_{{pass}} = {p_p_bot:.2f}$ kPa")
    tex.append(r"\end{itemize}")

    # 2. Soil Vertical weight and Surcharge
    tex.append(r"\subsubsection{Vertical Overburden \& Surcharge pressures}")
    p_v = q_soil + q
    tex.append(r"\begin{itemize}")
    tex.append(f"\\item Footing heel total vertical surcharge + backfill pressure ($p_{{v,heel}}$): $q_{{soil}} + q = {p_v:.2f}$ kPa")
    tex.append(r"\end{itemize}")

    # 3. Hydrostatic and uplift
    tex.append(r"\subsubsection{Hydrostatic Water \& Uplift Pressures}")
    p_w_back = gamma_w * Hwtr
    p_w_front = gamma_w * Hwtr_front
    p_upl = gamma_w * Hwtr
    tex.append(r"\begin{itemize}")
    tex.append(f"\\item Backwater hydrostatic pressure at footing base: $\\gamma_w \\cdot H_{{wtr}} = {p_w_back:.2f}$ kPa")
    tex.append(f"\\item Front water hydrostatic pressure at footing base: $\\gamma_w \\cdot H_{{wtr,front}} = {p_w_front:.2f}$ kPa")
    tex.append(f"\\item Footing base uniform water uplift pressure: $\\gamma_w \\cdot H_{{wtr}} = {p_upl:.2f}$ kPa")
    tex.append(r"\end{itemize}")

    # Section 4: Stability Report
    tex.append(r"\newpage")
    tex.append(r"\section{Stability Analysis \& Verification}")
    tex.append(r"The calculations for factors of safety against overturning, sliding, and slope stability are summarized below:")
    
    tex.append(r"\subsection{Sliding Safety Factor Check}")
    tex.append(r"The safety factor against sliding is calculated as the ratio of resisting shear forces along the footing base (including mobilized passive soil resistance) to driving active lateral forces:")
    tex.append(r"\begin{equation}")
    tex.append(r"FS_{sliding} = \frac{\sum R_{resisting}}{\sum F_{driving}} = \frac{R_{sliding} + P_{passive}}{F_{driving}}")
    tex.append(r"\end{equation}")
    if 'FS_slide_global' in globals() and FS_slide_global is not None:
        tex.append(f"Calculated Sliding FS = {FS_slide_global:.3f} (Required $\\ge 1.5$) \\hfill [{'PASS' if FS_slide_global >= 1.5 else 'FAIL'}]")

    tex.append(r"\subsection{Overturning Safety Factor Check}")
    tex.append(r"The safety factor against overturning is computed about the toe edge pivot point ($x=0.0$) as the ratio of stabilizing resisting moments to overturning driving moments:")
    tex.append(r"\begin{equation}")
    tex.append(r"FS_{overturning} = \frac{\sum M_{resisting}}{\sum M_{overturning}}")
    tex.append(r"\end{equation}")
    if 'FS_ot_global' in globals() and FS_ot_global is not None:
        tex.append(f"Calculated Overturning FS = {FS_ot_global:.3f} (Required $\\ge 1.5$) \\hfill [{'PASS' if FS_ot_global >= 1.5 else 'FAIL'}]")
    
    if 'fs_ssrm' in globals() and fs_ssrm is not None:
        tex.append(r"\subsection{Slope Stability SSRM Safety Factor Check}")
        tex.append(f"Global SSRM Slope Stability: Actual FS = {fs_ssrm:.3f} \\hfill [{'PASS' if ssrm_pass else 'FAIL'}]")
    
    if 'fig_ssrm' in globals() and fig_ssrm is not None:
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.8\textwidth]{report_temp/stability_ssrm.png}")
        tex.append(r"\caption{SSRM Viscoplastic Shear Strain (Failure Surface) Diagram}")
        tex.append(r"\end{figure}")
        
    # Section 5: Bearing Capacity Calculation Breakdown
    if enable_bearing:
        tex.append(r"\newpage")
        tex.append(r"\section{Bearing Capacity Breakdown}")
        tex.append(r"The bearing capacity safety factor and consolidation settlements are verified as follows:")
        
        tex.append(r"\subsection{Bearing Capacity & Eccentricity check}")
        tex.append(r"The maximum and minimum soil contact stresses are calculated by checking structural eccentricity ($e$):")
        tex.append(r"\begin{equation}")
        tex.append(r"\sigma_{max,min} = \frac{R_v}{B} \left( 1 \pm \frac{6e}{B} \right) \quad \text{for } e \le B/6")
        tex.append(r"\end{equation}")
        tex.append(r"\begin{equation}")
        tex.append(r"FS_{bearing} = \frac{q_{ult}}{\sigma_{max}}")
        tex.append(r"\end{equation}")
        
        tex.append(r"\begin{itemize}")
        if 'q_ult' in globals() and q_ult is not None:
            tex.append(f"\\item Ultimate bearing capacity: $q_{{ult}}$ = {q_ult:.2f} kPa")
            if 'sigma_max' in globals() and sigma_max is not None:
                tex.append(f"\\item Max bearing stress: $\\sigma_{{max}}$ = {sigma_max:.2f} kPa")
                tex.append(f"\\item Safety factor against bearing failure: Actual FS = {q_ult/sigma_max:.2f} (Required $\\ge {FS_bearing:.1f}$)")
        tex.append(r"\end{itemize}")

        tex.append(r"\subsection{Consolidation Settlement Calculation}")
        tex.append(r"One-dimensional primary consolidation settlement is computed using the compression index $C_c$, thickness $H_c$, and vertical stress increase:")
        tex.append(r"\begin{equation}")
        tex.append(r"S_c = \frac{C_c \cdot H_c}{1 + e_0} \log\left( \frac{\sigma'_{v0} + \Delta \sigma}{\sigma'_{v0}} \right)")
        tex.append(r"\end{equation}")
        tex.append(r"\begin{itemize}")
        if 'consol_settlement_mm' in globals() and consol_settlement_mm is not None:
            tex.append(f"\\item Consolidation settlement: $s_{{consol}}$ = {consol_settlement_mm:.2f} mm (Allowable = {consol_allow_mm:.2f} mm) \\hfill [{'PASS' if consol_pass else 'FAIL'}]")
        tex.append(r"\end{itemize}")

    # Section 6: FEM Settlement and Displacement Report
    tex.append(r"\newpage")
    tex.append(r"\section{FEM Settlement \& Bottom Displacements}")
    tex.append(r"Deformations and settlements computed at the base of the footing using soil spring finite element analysis:")
    tex.append(r"\begin{itemize}")
    tex.append(f"\\item Toe settlement: {settlement_toe_mm:.2f} mm")
    tex.append(f"\\item Heel settlement: {settlement_heel_mm:.2f} mm")
    tex.append(f"\\item Differential settlement: {diff_settlement_mm:.2f} mm (Allowable $\\le {allowable_diff_settlement_mm:.1f}$ mm) \\hfill [{'PASS' if diff_settlement_pass else 'FAIL'}]")
    tex.append(f"\\item Wall rotation: {rotation_deg:.4f}$^\\circ$")
    tex.append(r"\end{itemize}")

    if 'df_bottom_disp' in globals() and df_bottom_disp is not None and not df_bottom_disp.empty:
        tex.append(r"\subsection{Footing Base Nodes Displacement Table}")
        tex.append(r"\begin{longtable}{ccccc}")
        tex.append(r"\toprule")
        tex.append(r"Node & X Position (m) & Disp X (mm) & Disp Y (mm) & Settlement (mm) \\")
        tex.append(r"\midrule")
        for idx, row in df_bottom_disp.iterrows():
            tex.append(f"{int(row['Node'])} & {row['X Position (m)']:.3f} & {row['Disp X (mm)']:.4f} & {row['Disp Y (mm)']:.4f} & {row['Settlement (mm)']:.4f} \\\\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{longtable}")

    # Section: Finite Element Stress contours and Bending moment plots
    tex.append(r"\newpage")
    tex.append(r"\section{Finite Element Stress \& Bending Moment Contours}")
    tex.append(r"Finite element stress contours and internal bending moments mapping across the concrete wall cross-section:")
    
    if os.path.exists("report_temp/stem_bending_moment.png"):
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.75\textwidth]{report_temp/stem_bending_moment.png}")
        tex.append(r"\caption{Stem Wall Internal Bending Moment Diagram (kN$\cdot$m)}")
        tex.append(r"\end{figure}")
        
    if os.path.exists("report_temp/stress_xx.png"):
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.7\textwidth]{report_temp/stress_xx.png}")
        tex.append(r"\caption{FE stress contour in X-Direction $\sigma_{xx}$ (kN/m$^2$)}")
        tex.append(r"\end{figure}")

    if os.path.exists("report_temp/stress_yy.png"):
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.7\textwidth]{report_temp/stress_yy.png}")
        tex.append(r"\caption{FE stress contour in Y-Direction $\sigma_{yy}$ (kN/m$^2$)}")
        tex.append(r"\end{figure}")

    # Section 7: Pile Design
    if enable_pile:
        tex.append(r"\newpage")
        tex.append(r"\section{Pile Foundation Design \& Verification}")
        tex.append(r"Detailed pile demands, capacities, Winkler analysis results, and interaction diagram checks:")
        
        # Demands vs Capacities
        tex.append(r"\subsection{Axial \& Lateral Pile Capacities}")
        tex.append(r"\begin{longtable}{lll}")
        tex.append(r"\toprule")
        tex.append(r"Capacity Parameter & Symbol & Value \\")
        tex.append(r"\midrule")
        if 'Q_comp_allow' in globals() and Q_comp_allow is not None:
            tex.append(f"Allowable Compressive Capacity & $Q_{{comp,allow}}$ & {Q_comp_allow:.2f} kN \\\\")
            tex.append(f"Allowable Tensile Capacity & $Q_{{tens,allow}}$ & {Q_tens_allow:.2f} kN \\\\")
        if 'H_allow' in globals() and H_allow is not None:
            tex.append(f"Allowable Lateral Capacity & $H_{{allow}}$ & {H_allow:.2f} kN \\\\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{longtable}")

        # Verification checklist
        tex.append(r"\subsection{Pile Foundation Demands Verification}")
        tex.append(r"\begin{itemize}")
        if 'P_toe' in globals() and P_toe is not None:
            tex.append(f"\\item \\textbf{{Toe Pile Axial Demand}}: $P_{{toe}}$ = {P_toe:.2f} kN \\hfill [{'PASS' if pile_pass_toe_axial else 'FAIL'}]")
            tex.append(f"\\item \\textbf{{Toe Pile Lateral Demand}}: $V_{{max,toe}}$ = {V_max_toe:.2f} kN \\hfill [{'PASS' if pile_pass_toe_lateral else 'FAIL'}]")
            tex.append(f"\\item \\textbf{{Toe Pile Interaction Check}}: \\hfill [{'PASS' if pile_pass_toe_interaction else 'FAIL'}]")
            
        if 'P_heel' in globals() and P_heel is not None:
            tex.append(f"\\item \\textbf{{Heel Pile Axial Demand}}: $P_{{heel}}$ = {P_heel:.2f} kN \\hfill [{'PASS' if pile_pass_heel_axial else 'FAIL'}]")
            tex.append(f"\\item \\textbf{{Heel Pile Lateral Demand}}: $V_{{max,heel}}$ = {V_max_heel:.2f} kN \\hfill [{'PASS' if pile_pass_heel_lateral else 'FAIL'}]")
            tex.append(f"\\item \\textbf{{Heel Pile Interaction Check}}: \\hfill [{'PASS' if pile_pass_heel_interaction else 'FAIL'}]")
        tex.append(r"\end{itemize}")

        # Pile Drawings
        tex.append(r"\subsection{Pile Layout \& Elevation Schematic}")
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.6\textwidth]{report_temp/pile_elevation.png}")
        tex.append(r"\caption{Pile Layout under Footing Base with Reaction Forces}")
        tex.append(r"\end{figure}")
        
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.8\textwidth]{report_temp/pile_winkler.png}")
        tex.append(r"\caption{Pile Winkler Analysis Response Profiles}")
        tex.append(r"\end{figure}")
        
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.8\textwidth]{report_temp/pile_interaction.png}")
        tex.append(r"\caption{Moment-Axial capacity envelope of the Pile}")
        tex.append(r"\end{figure}")
        
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(r"\includegraphics[width=0.5\textwidth]{report_temp/pile_section.png}")
        tex.append(r"\caption{Pile Cross Section Details}")
        tex.append(r"\end{figure}")

    tex.append(r"\end{document}")
    return "\n".join(tex)

def generate_pdf_reportlab(output_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    import os
    import datetime

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#162447'),
        alignment=1, # Center
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading2'],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1f4068'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading3'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1f4068'),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8
    )

    story = []
    
    # Title Page
    story.append(Spacer(1, 40))
    story.append(Paragraph("Retaining Wall Design & Verification Report", title_style))
    story.append(Paragraph("<b>RT Wall Cantilever</b>", ParagraphStyle('SubTitle', parent=body_style, alignment=1, fontSize=11)))
    story.append(Paragraph(f"Date: {datetime.date.today().strftime('%B %d, %Y')}", ParagraphStyle('DateStyle', parent=body_style, alignment=1)))
    story.append(Spacer(1, 40))
    story.append(Paragraph("This engineering report compiles the retaining wall dimensions, material properties, applied loads, stability safety factors, bearing capacity analysis, FEM settlement profiles, and pile foundation design checks.", body_style))
    story.append(PageBreak())
    
    # 1. Inputs
    story.append(Paragraph("1. Design Inputs & Parameters", h1_style))
    story.append(Paragraph("<b>Wall Geometry:</b>", h2_style))
    
    geom_data = [
        ["Parameter", "Symbol", "Value"],
        ["Out-of-plane thickness", Paragraph("<i>t</i>", body_style), f"{t:.3f} m"],
        ["Wall height", Paragraph("<i>H<sub>w</sub></i>", body_style), f"{Hw:.3f} m"],
        ["Toe length", Paragraph("<i>L<sub>toe</sub></i>", body_style), f"{toe:.3f} m"],
        ["Heel length", Paragraph("<i>L<sub>heel</sub></i>", body_style), f"{heel:.3f} m"],
        ["Top wall thickness", Paragraph("<i>b<sub>top</sub></i>", body_style), f"{top_wall:.3f} m"],
        ["Bottom wall thickness", Paragraph("<i>b<sub>bot</sub></i>", body_style), f"{bot_wall:.3f} m"],
        ["Footing thickness", Paragraph("<i>h<sub>ftg</sub></i>", body_style), f"{h_ftg:.3f} m"]
    ]
    if include_shear_key:
        geom_data.append(["Shear key distance", Paragraph("<i>x<sub>sk</sub></i>", body_style), f"{shear_key_distance:.3f} m"])
        geom_data.append(["Shear key width", Paragraph("<i>w<sub>sk</sub></i>", body_style), f"{shear_key_width:.3f} m"])
        geom_data.append(["Shear key thickness", Paragraph("<i>t<sub>sk</sub></i>", body_style), f"{shear_key_thickness:.3f} m"])
        
    t_geom = Table(geom_data, colWidths=[200, 100, 150])
    t_geom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e1f5fe')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0277bd')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(t_geom)
    story.append(Spacer(1, 10))
    
    # SNI Geotechnical Geometry checklist table
    H_tot = Hw + h_ftg
    story.append(Paragraph("<b>SNI Typical Geometry Verification Checklist:</b>", h2_style))
    slope_val = taper / Hw if (Hw > 0 and taper_direction == 'Toe-facing') else 0.0
    slope_status = "PASS" if (taper_direction == 'Heel-facing' or slope_val >= (1.0/48.0)) else "FAIL"
    slope_req = ">= 0.0208" if taper_direction == 'Toe-facing' else "N/A (Vertical)"
    
    sni_data = [
        ["Parameter Name", "Symbol", "Value", "SNI Requirement", "Status"],
        ["Top Wall Thickness", Paragraph("<i>b<sub>top</sub></i>", body_style), f"{top_wall:.2f} m", ">= 0.30 m", "PASS" if top_wall >= 0.30 else "FAIL"],
        ["Base Stem Thickness", Paragraph("<i>b<sub>bot</sub></i>", body_style), f"{bot_wall:.2f} m", f">= {0.1*H_tot:.2f} m", "PASS" if bot_wall >= 0.1*H_tot else "FAIL"],
        ["Front Batter Slope", Paragraph("<i>slope</i>", body_style), f"{slope_val:.4f}", slope_req, slope_status],
        ["Footing Width", Paragraph("<i>B</i>", body_style), f"{ftg:.2f} m", f"{0.4*H_tot:.2f} ~ {0.7*H_tot:.2f} m", "PASS" if 0.4*H_tot <= ftg <= 0.7*H_tot else "FAIL"],
        ["Footing Thickness", Paragraph("<i>h<sub>ftg</sub></i>", body_style), f"{h_ftg:.2f} m", f"{H_tot/12:.2f} ~ {H_tot/10:.2f} m", "PASS" if H_tot/12 <= h_ftg <= H_tot/10 else "FAIL"],
        ["Toe Slab Length", Paragraph("<i>L<sub>toe</sub></i>", body_style), f"{toe:.2f} m", f">= {ftg/3:.2f} m", "PASS" if toe >= ftg/3 else "FAIL"]
    ]
    t_sni = Table(sni_data, colWidths=[150, 60, 80, 110, 50])
    t_sni.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#efebe9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(t_sni)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Soil & Water Parameters:</b>", h2_style))
    soil_data = [
        ["Parameter", "Symbol", "Value"],
        ["Dry unit weight", Paragraph("<i>&gamma;<sub>dry</sub></i>", body_style), f"{gamma_soil_dry:.2f} kN/m³"],
        ["Wet unit weight", Paragraph("<i>&gamma;<sub>wet</sub></i>", body_style), f"{gamma_soil_wet:.2f} kN/m³"],
        ["Friction angle", Paragraph("<i>&phi;</i>", body_style), f"{phi:.1f}°"],
        ["Soil cohesion", Paragraph("<i>c</i>", body_style), f"{c_soil:.1f} kPa"],
        ["Surcharge load", Paragraph("<i>q</i>", body_style), f"{q:.2f} kPa"],
        ["Soil height above heel", Paragraph("<i>h<sub>soil</sub></i>", body_style), f"{h_soil:.3f} m"],
        ["Soil above toe", Paragraph("<i>h<sub>toe</sub></i>", body_style), f"{h_soil_toe:.3f} m"],
        ["Water unit weight", Paragraph("<i>&gamma;<sub>w</sub></i>", body_style), f"{gamma_w:.2f} kN/m³"],
        ["Water height behind wall", Paragraph("<i>H<sub>wtr</sub></i>", body_style), f"{Hwtr:.2f} m"],
        ["Water height in front", Paragraph("<i>H<sub>wtr,front</sub></i>", body_style), f"{Hwtr_front:.3f} m"]
    ]
    t_soil = Table(soil_data, colWidths=[200, 100, 150])
    t_soil.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f8e9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#33691e')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(t_soil)
    story.append(PageBreak())
    
    # 2. Geometries image
    story.append(Paragraph("2. Model & Geometry Diagrams", h1_style))
    if os.path.exists("report_temp/model_geometry.png"):
        story.append(Paragraph("<b>Wall Profile & Mesh Grid:</b>", h2_style))
        story.append(Image("report_temp/model_geometry.png", width=450, height=280))
        story.append(Spacer(1, 10))
    if os.path.exists("report_temp/typical_dimensions.png"):
        story.append(Paragraph("<b>SNI Geotechnical Dimensions Proportions Verification:</b>", h2_style))
        story.append(Image("report_temp/typical_dimensions.png", width=450, height=280))
    story.append(PageBreak())
    
    # 3. Loads
    story.append(Paragraph("3. Applied Loads & Pressure Distributions", h1_style))
    story.append(Paragraph("The pressure diagrams and corresponding analytical safety check calculations are shown below:", body_style))
    
    # Calculations
    h_active = min(Hw, h_soil)
    if surcharge_type == 'Strip Load':
        q_top = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - h_active))['delta sigma x [kPa]'] if h_soil > 0 else 0.0
        q_bot = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
    else:
        q_top = Ka * q if h_soil > 0 else 0.0
        q_bot = Ka * q if h_soil > 0 else 0.0
    p_top_act = Ka * (gamma_soil_dry * max(0.0, h_soil - h_active)) + q_top if h_soil > 0 else 0.0
    p_bot_act = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot if h_soil > 0 else 0.0
    
    FS_passive_val = 2.0
    Kp_val = np.tan(np.radians(45.0 + phi / 2.0)) ** 2
    p_p_top = (Kp_val * gamma_soil_dry * h_soil_toe) / FS_passive_val if h_soil_toe > 0 else 0.0
    p_p_bot = (Kp_val * gamma_soil_dry * (h_soil_toe + h_ftg)) / FS_passive_val
    
    p_v = q_soil + q
    p_w_back = gamma_w * Hwtr
    p_w_front = gamma_w * Hwtr_front
    p_upl = gamma_w * Hwtr

    # Soil Lateral Pressure (Active & Passive)
    if os.path.exists("report_temp/load_soil_lateral.png"):
        story.append(Paragraph("<b>A. Soil Lateral Pressure (Active & Passive):</b>", h2_style))
        story.append(Image("report_temp/load_soil_lateral.png", width=380, height=230))
        story.append(Spacer(1, 5))
        
        story.append(Paragraph("• <b>Active Earth Pressure Coefficient Equation:</b>", body_style))
        if os.path.exists("report_temp/eq_ka.png"):
            story.append(Image("report_temp/eq_ka.png", width=220, height=33))
        story.append(Paragraph(f"Result: <i>K<sub>a</sub></i> = {Ka:.4f}", body_style))
        story.append(Spacer(1, 5))
        
        story.append(Paragraph("• <b>Active Pressure (Top of Stem):</b>", body_style))
        if os.path.exists("report_temp/eq_p_act_top.png"):
            story.append(Image("report_temp/eq_p_act_top.png", width=180, height=30))
        story.append(Paragraph(f"Result: <i>p<sub>act,top</sub></i> = {p_top_act:.2f} kPa", body_style))
        story.append(Spacer(1, 5))

        story.append(Paragraph("• <b>Active Pressure (Bottom of Footing):</b>", body_style))
        if os.path.exists("report_temp/eq_p_act_bot.png"):
            story.append(Image("report_temp/eq_p_act_bot.png", width=340, height=30))
        story.append(Paragraph(f"Result: <i>p<sub>act,bot</sub></i> = {p_bot_act:.2f} kPa", body_style))
        story.append(Spacer(1, 5))

        story.append(Paragraph("• <b>Passive Earth Pressure Coefficient Equation:</b>", body_style))
        if os.path.exists("report_temp/eq_kp.png"):
            story.append(Image("report_temp/eq_kp.png", width=220, height=33))
        story.append(Paragraph(f"Result: <i>K<sub>p</sub></i> = {Kp_val:.4f}", body_style))
        story.append(Spacer(1, 5))

        story.append(Paragraph("• <b>Passive Pressure (Bottom of Footing):</b>", body_style))
        if os.path.exists("report_temp/eq_p_pass_bot.png"):
            story.append(Image("report_temp/eq_p_pass_bot.png", width=260, height=33))
        story.append(Paragraph(f"Result: <i>p<sub>pass,bot</sub></i> = {p_p_bot:.2f} kPa (Mobilized, FS_pass = 2.0)", body_style))
        story.append(Spacer(1, 10))

    # Soil Vertical Weight & Surcharge
    if os.path.exists("report_temp/load_soil_vertical.png"):
        story.append(Paragraph("<b>B. Soil Vertical Weight & Surcharge:</b>", h2_style))
        story.append(Image("report_temp/load_soil_vertical.png", width=380, height=230))
        story.append(Spacer(1, 5))
        story.append(Paragraph("• <b>Vertical Backfill Pressure Equation:</b>", body_style))
        if os.path.exists("report_temp/eq_p_v_heel.png"):
            story.append(Image("report_temp/eq_p_v_heel.png", width=180, height=30))
        story.append(Paragraph(f"Result: <i>p<sub>v,heel</sub></i> = {p_v:.2f} kPa", body_style))
        story.append(Spacer(1, 10))

    # Hydrostatic Water Pressure
    if os.path.exists("report_temp/load_hydrostatic.png"):
        story.append(Paragraph("<b>C. Hydrostatic Water Pressure:</b>", h2_style))
        story.append(Image("report_temp/load_hydrostatic.png", width=380, height=230))
        story.append(Spacer(1, 5))
        story.append(Paragraph("• <b>Hydrostatic Water Pressure Equation:</b>", body_style))
        if os.path.exists("report_temp/eq_p_w.png"):
            story.append(Image("report_temp/eq_p_w.png", width=140, height=30))
        story.append(Paragraph(f"Result: Backwater <i>p<sub>w,back</sub></i> = {p_w_back:.2f} kPa, Frontwater <i>p<sub>w,front</sub></i> = {p_w_front:.2f} kPa", body_style))
        story.append(Spacer(1, 10))

    # Uplift Water Pressure under Base
    if os.path.exists("report_temp/load_uplift.png"):
        story.append(Paragraph("<b>D. Uplift Water Pressure under Base:</b>", h2_style))
        story.append(Image("report_temp/load_uplift.png", width=380, height=230))
        story.append(Spacer(1, 5))
        story.append(Paragraph("• <b>Base Uplift Pressure Equation:</b>", body_style))
        if os.path.exists("report_temp/eq_p_uplift.png"):
            story.append(Image("report_temp/eq_p_uplift.png", width=180, height=30))
        story.append(Paragraph(f"Result: <i>p<sub>uplift</sub></i> = {p_upl:.2f} kPa", body_style))
        story.append(Spacer(1, 10))

    # Pseudo-Static Seismic Forces
    if os.path.exists("report_temp/load_seismic.png"):
        story.append(Paragraph("<b>E. Pseudo-Static Seismic Forces:</b>", h2_style))
        story.append(Image("report_temp/load_seismic.png", width=380, height=230))
        story.append(Spacer(1, 10))

    # Total / Combined Load System
    if os.path.exists("report_temp/load_total.png"):
        story.append(Paragraph("<b>F. Total / Combined Load System:</b>", h2_style))
        story.append(Image("report_temp/load_total.png", width=380, height=230))
        story.append(Spacer(1, 10))
            
    story.append(PageBreak())
    
    # 4. Stability report
    story.append(Paragraph("4. Stability Verification", h1_style))
    story.append(Paragraph("Safety factor analysis for overall retaining wall sliding, overturning, and slope failures:", body_style))
    
    # Equations
    story.append(Paragraph("<b>Stability Verification Equations:</b>", h2_style))
    story.append(Paragraph("• Sliding safety factor equation:", body_style))
    if os.path.exists("report_temp/eq_sliding.png"):
        story.append(Image("report_temp/eq_sliding.png", width=340, height=33))
    story.append(Spacer(1, 5))
    story.append(Paragraph("• Overturning safety factor equation:", body_style))
    if os.path.exists("report_temp/eq_overturning.png"):
        story.append(Image("report_temp/eq_overturning.png", width=260, height=33))
    story.append(Spacer(1, 10))

    stab_items = []
    if 'FS_slide_global' in globals() and FS_slide_global is not None:
        pass_slide = "PASS" if FS_slide_global >= 1.5 else "FAIL"
        stab_items.append([Paragraph("Sliding Safety Factor Check", body_style), f"FS = {FS_slide_global:.3f}", f"Req >= 1.500", pass_slide])
    if 'FS_ot_global' in globals() and FS_ot_global is not None:
        pass_ot = "PASS" if FS_ot_global >= 1.5 else "FAIL"
        stab_items.append([Paragraph("Overturning Safety Factor Check", body_style), f"FS = {FS_ot_global:.3f}", f"Req >= 1.500", pass_ot])
    if 'fs_ssrm' in globals() and fs_ssrm is not None:
        pass_ssrm = "PASS" if ssrm_pass else "FAIL"
        stab_items.append([Paragraph("Slope Stability SSRM Safety Factor Check", body_style), f"FS = {fs_ssrm:.3f}", f"Req >= {1.1 if (PGA * FPGA) > 0 else 1.5}", pass_ssrm])
        
    t_stab = Table(stab_items, colWidths=[200, 100, 100, 50])
    t_stab.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(t_stab)
    story.append(Spacer(1, 15))
    
    if os.path.exists("report_temp/xslope_inputs.png"):
        story.append(Paragraph("<b>XSLOPE Slope Geometry & Inputs Diagram:</b>", h2_style))
        story.append(Image("report_temp/xslope_inputs.png", width=450, height=200))
        story.append(Spacer(1, 10))
    if os.path.exists("report_temp/xslope_mesh.png"):
        story.append(Paragraph("<b>XSLOPE FEM Mesh with Material Zones (tri6):</b>", h2_style))
        story.append(Image("report_temp/xslope_mesh.png", width=450, height=200))
        story.append(Spacer(1, 10))
    if os.path.exists("report_temp/stability_ssrm.png"):
        story.append(Paragraph("<b>2D SSRM Viscoplastic Shear Strain Failure Surface:</b>", h2_style))
        story.append(Image("report_temp/stability_ssrm.png", width=450, height=200))
        story.append(Spacer(1, 10))
        
    story.append(PageBreak())
    
    # 5. Bearing Capacity Breakdown (if checked)
    if enable_bearing:
        story.append(Paragraph("5. Bearing Capacity & Consolidation Settlement", h1_style))
        story.append(Paragraph("Footing geotechnical bearing capacity design check and consolidation settlement breakdown:", body_style))
        
        # Equations
        story.append(Paragraph("<b>Bearing Stress & Settlement Equations:</b>", h2_style))
        story.append(Paragraph("• Bearing contact stress:", body_style))
        if os.path.exists("report_temp/eq_bearing_stress.png"):
            story.append(Image("report_temp/eq_bearing_stress.png", width=250, height=33))
        story.append(Spacer(1, 5))
        story.append(Paragraph("• Bearing safety factor:", body_style))
        if os.path.exists("report_temp/eq_bearing_fs.png"):
            story.append(Image("report_temp/eq_bearing_fs.png", width=180, height=33))
        story.append(Spacer(1, 5))
        story.append(Paragraph("• Consolidation settlement:", body_style))
        if os.path.exists("report_temp/eq_settlement.png"):
            story.append(Image("report_temp/eq_settlement.png", width=280, height=33))
        story.append(Spacer(1, 10))

        bearing_items = []
        if 'q_ult' in globals() and q_ult is not None:
            bearing_items.append(["Ultimate soil bearing capacity (q_ult)", f"{q_ult:.2f} kPa"])
            if 'sigma_max' in globals() and sigma_max is not None:
                bearing_items.append(["Max bearing stress (sigma_max)", f"{sigma_max:.2f} kPa"])
                bearing_items.append(["Bearing capacity Safety Factor", f"FS = {q_ult/sigma_max:.2f} (Req >= {FS_bearing:.1f})"])
        if 'consol_settlement_mm' in globals() and consol_settlement_mm is not None:
            pass_c = "PASS" if consol_pass else "FAIL"
            bearing_items.append(["Consolidation Settlement", f"{consol_settlement_mm:.2f} mm (Allowable: {consol_allow_mm:.2f} mm) -> {pass_c}"])
            
        t_bear = Table(bearing_items, colWidths=[300, 150])
        t_bear.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t_bear)
        story.append(Spacer(1, 15))
        
    # 6. FEM Settlement
    story.append(Paragraph("6. Finite Element Settlement & Base Displacements", h1_style))
    story.append(Paragraph(f" Toe Settlement: <b>{settlement_toe_mm:.2f} mm</b>", body_style))
    story.append(Paragraph(f" Heel Settlement: <b>{settlement_heel_mm:.2f} mm</b>", body_style))
    pass_diff = "PASS" if diff_settlement_pass else "FAIL"
    story.append(Paragraph(f" Differential Settlement: <b>{diff_settlement_mm:.2f} mm</b> (Allowable: 50.00 mm) -> <b>{pass_diff}</b>", body_style))
    story.append(Paragraph(f" Rotational Tilt: <b>{rotation_deg:.4f}°</b>", body_style))
    story.append(Spacer(1, 10))
    
    if 'df_bottom_disp' in globals() and df_bottom_disp is not None and not df_bottom_disp.empty:
        story.append(Paragraph("<b>Footing Base Nodes Settlement Data:</b>", h2_style))
        disp_headers = [["Node", "X Position (m)", "Disp X (mm)", "Disp Y (mm)", "Settlement (mm)"]]
        for idx, row in df_bottom_disp.iterrows():
            disp_headers.append([
                str(int(row['Node'])),
                f"{row['X Position (m)']:.3f}",
                f"{row['Disp X (mm)']:.4f}",
                f"{row['Disp Y (mm)']:.4f}",
                f"{row['Settlement (mm)']:.4f}"
            ])
        t_disp = Table(disp_headers, colWidths=[60, 100, 90, 90, 110])
        t_disp.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eceff1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('ALIGN', (0,0), (-1,-1), 'CENTER')
        ]))
        story.append(t_disp)

    # 7. FE Stress and Stem Bending Moment plots
    story.append(PageBreak())
    story.append(Paragraph("7. Finite Element Stress & Bending Moment Contours", h1_style))
    story.append(Paragraph("2D internal bending moments and finite element stress contours plotted on the retaining wall concrete cross-section:", body_style))
    story.append(Spacer(1, 10))
    if os.path.exists("report_temp/stem_bending_moment.png"):
        story.append(Paragraph("<b>Stem Node Bending Moment Diagram (kN·m):</b>", h2_style))
        story.append(Image("report_temp/stem_bending_moment.png", width=420, height=270))
        story.append(Spacer(1, 15))
    if os.path.exists("report_temp/stress_xx.png"):
        story.append(Paragraph("<b>FE Stress Contour σ_xx (kN/m²):</b>", h2_style))
        story.append(Image("report_temp/stress_xx.png", width=400, height=270))
        story.append(Spacer(1, 15))
    if os.path.exists("report_temp/stress_yy.png"):
        story.append(Paragraph("<b>FE Stress Contour σ_yy (kN/m²):</b>", h2_style))
        story.append(Image("report_temp/stress_yy.png", width=400, height=270))
        
    # 8. Pile foundation
    if enable_pile:
        story.append(PageBreak())
        story.append(Paragraph("8. Pile Foundation Design & Verification", h1_style))
        story.append(Paragraph("Geotechnical capacity, design parameters, and structural envelopes of the foundation piles:", body_style))
        
        story.append(Paragraph("<b>Pile Cross Section Geometry & Material Details:</b>", h2_style))
        pile_param_data = [
            ["Parameter Name", "Symbol", "Value"],
            ["Pile Material Type", "-", pile_material],
            ["Pile Cross-section Shape", "-", pile_shape],
        ]
        if pile_material == 'Concrete':
            if pile_shape == 'Circle':
                pile_param_data.append(["Pile Diameter", "d_pile", f"{diameter_pile:.3f} m"])
                pile_param_data.append(["Concrete Cover", "d_c", f"{cover_pile:.1f} mm"])
                pile_param_data.append(["Main Reinforcement", "-", f"{int(n_rebar_pile)} D{rebar_dia_pile}"])
            else:
                pile_param_data.append(["Pile Dimensions (dx × dy)", "h_pile × b_pile", f"{width_x_pile:.3f} m × {width_y_pile:.3f} m"])
                pile_param_data.append(["Concrete Cover", "d_c", f"{cover_pile:.1f} mm"])
                pile_param_data.append(["Main Reinforcement", "-", f"4 D{rebar_dia_pile} (Corners)"])
        else:
            if pile_shape == 'Circle':
                pile_param_data.append(["Pile Diameter", "d_pile", f"{diameter_pile:.3f} m"])
            else:
                pile_param_data.append(["Pile Dimensions (dx × dy)", "h_pile × b_pile", f"{width_x_pile:.3f} m × {width_y_pile:.3f} m"])
                
        t_pile_params = Table(pile_param_data, colWidths=[200, 100, 150])
        t_pile_params.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eceff1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_pile_params)
        story.append(Spacer(1, 10))
        
        if os.path.exists("report_temp/pile_section.png"):
            story.append(Paragraph("<b>Pile Concrete Section Drawing:</b>", h2_style))
            story.append(Image("report_temp/pile_section.png", width=250, height=200))
            story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Pile Structural Capacity Envelope limits:</b>", h2_style))
        cap_items = [
            ["Capacity Metric", "Allowable Value"],
            ["Axial Compression Capacity", f"{Q_comp_allow:.2f} kN" if 'Q_comp_allow' in globals() else "N/A"],
            ["Axial Tension Capacity", f"{Q_tens_allow:.2f} kN" if 'Q_tens_allow' in globals() else "N/A"],
            ["Lateral Shear Capacity", f"{H_allow:.2f} kN" if 'H_allow' in globals() else "N/A"]
        ]
        t_cap = Table(cap_items, colWidths=[250, 200])
        t_cap.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eceff1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_cap)
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Demands Verification checklist:</b>", h2_style))
        story.append(Paragraph(f" Toe Pile Axial demand: <b>{P_toe:.2f} kN</b> -> <b>{'PASS' if pile_pass_toe_axial else 'FAIL'}</b>" if 'P_toe' in globals() else "", body_style))
        story.append(Paragraph(f" Toe Pile Lateral shear demand: <b>{V_max_toe:.2f} kN</b> -> <b>{'PASS' if pile_pass_toe_lateral else 'FAIL'}</b>" if 'V_max_toe' in globals() else "", body_style))
        story.append(Paragraph(f" Toe Pile Interaction Envelope check: <b>{'PASS' if pile_pass_toe_interaction else 'FAIL'}</b>" if 'pile_pass_toe_interaction' in globals() else "", body_style))
        story.append(Spacer(1, 5))
        story.append(Paragraph(f" Heel Pile Axial demand: <b>{P_heel:.2f} kN</b> -> <b>{'PASS' if pile_pass_heel_axial else 'FAIL'}</b>" if 'P_heel' in globals() else "", body_style))
        story.append(Paragraph(f" Heel Pile Lateral shear demand: <b>{V_max_heel:.2f} kN</b> -> <b>{'PASS' if pile_pass_heel_lateral else 'FAIL'}</b>" if 'V_max_heel' in globals() else "", body_style))
        story.append(Paragraph(f" Heel Pile Interaction Envelope check: <b>{'PASS' if pile_pass_heel_interaction else 'FAIL'}</b>" if 'pile_pass_heel_interaction' in globals() else "", body_style))
        
        story.append(PageBreak())
        if os.path.exists("report_temp/pile_elevation.png"):
            story.append(Paragraph("<b>Pile Foundation Schematic under Base:</b>", h2_style))
            story.append(Image("report_temp/pile_elevation.png", width=380, height=310))
            story.append(Spacer(1, 10))
        if os.path.exists("report_temp/pile_winkler.png"):
            story.append(Paragraph("<b>Pile Winkler Analysis Elastic Profiles:</b>", h2_style))
            story.append(Image("report_temp/pile_winkler.png", width=450, height=300))
            
        story.append(PageBreak())
        if os.path.exists("report_temp/pile_interaction.png"):
            story.append(Paragraph("<b>Moment-Axial Interaction Envelope:</b>", h2_style))
            story.append(Image("report_temp/pile_interaction.png", width=380, height=310))
            story.append(Spacer(1, 10))
            
    doc.build(story)

def generate_docx_report(output_path):
    from docx import Document
    from docx.shared import Inches, Pt
    import os
    import datetime

    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    title_p = doc.add_paragraph()
    run = title_p.add_run("Retaining Wall Design & Verification Report\n")
    run.bold = True
    run.font.size = Pt(20)
    run.font.name = 'Arial'
    
    run_sub = title_p.add_run("RT Wall Cantilever\n")
    run_sub.italic = True
    run_sub.font.size = Pt(12)
    
    run_date = title_p.add_run(f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
    run_date.font.size = Pt(10)
    
    doc.add_paragraph("This engineering report compiles the retaining wall dimensions, material properties, applied loads, stability safety factors, bearing capacity analysis, FEM settlement profiles, and pile foundation design checks.")
    
    # 1. Inputs
    doc.add_heading("1. Design Inputs & Parameters", level=1)
    
    doc.add_heading("Wall Geometry", level=2)
    t_geom = doc.add_table(rows=1, cols=3)
    t_geom.style = 'Table Grid'
    hdr = t_geom.rows[0].cells
    hdr[0].text = 'Parameter'
    hdr[1].text = 'Symbol'
    hdr[2].text = 'Value'
    
    geom_items = [
        ("Out-of-plane thickness", "t", f"{t:.3f} m"),
        ("Wall height", "Hw", f"{Hw:.3f} m"),
        ("Toe length", "L_toe", f"{toe:.3f} m"),
        ("Heel length", "L_heel", f"{heel:.3f} m"),
        ("Top wall thickness", "b_top", f"{top_wall:.3f} m"),
        ("Bottom wall thickness", "b_bot", f"{bot_wall:.3f} m"),
        ("Footing thickness", "h_ftg", f"{h_ftg:.3f} m")
    ]
    if include_shear_key:
        geom_items.append(("Shear key distance", "x_sk", f"{shear_key_distance:.3f} m"))
        geom_items.append(("Shear key width", "w_sk", f"{shear_key_width:.3f} m"))
        geom_items.append(("Shear key thickness", "t_sk", f"{shear_key_thickness:.3f} m"))
        
    for p, s, v in geom_items:
        row = t_geom.add_row()
        row.cells[0].text = p
        row.cells[1].text = s
        row.cells[2].text = v
        
    # SNI checklist
    doc.add_heading("SNI Typical Geometry Verification Checklist", level=2)
    t_sni = doc.add_table(rows=1, cols=5)
    t_sni.style = 'Table Grid'
    hdr = t_sni.rows[0].cells
    hdr[0].text = 'Parameter Name'
    hdr[1].text = 'Symbol'
    hdr[2].text = 'Value'
    hdr[3].text = 'SNI Requirement'
    hdr[4].text = 'Status'
    
    H_tot = Hw + h_ftg
    slope_val = taper / Hw if (Hw > 0 and taper_direction == 'Toe-facing') else 0.0
    slope_status = "PASS" if (taper_direction == 'Heel-facing' or slope_val >= (1.0/48.0)) else "FAIL"
    slope_req = ">= 0.0208" if taper_direction == 'Toe-facing' else "N/A (Vertical)"
    sni_items = [
        ("Top Wall Thickness", "b_top", f"{top_wall:.2f} m", ">= 0.30 m", "PASS" if top_wall >= 0.30 else "FAIL"),
        ("Base Stem Thickness", "b_bot", f"{bot_wall:.2f} m", f">= {0.1*H_tot:.2f} m", "PASS" if bot_wall >= 0.1*H_tot else "FAIL"),
        ("Front Face Batter Slope", "slope", f"{slope_val:.4f}", slope_req, slope_status),
        ("Footing Width", "B", f"{ftg:.2f} m", f"{0.4*H_tot:.2f} ~ {0.7*H_tot:.2f} m", "PASS" if 0.4*H_tot <= ftg <= 0.7*H_tot else "FAIL"),
        ("Footing Thickness", "h_ftg", f"{h_ftg:.2f} m", f"{H_tot/12:.2f} ~ {H_tot/10:.2f} m", "PASS" if H_tot/12 <= h_ftg <= H_tot/10 else "FAIL"),
        ("Toe Slab Length", "L_toe", f"{toe:.2f} m", f">= {ftg/3:.2f} m", "PASS" if toe >= ftg/3 else "FAIL")
    ]
    for p, s, v, req_val, st_val in sni_items:
        row = t_sni.add_row()
        row.cells[0].text = p
        row.cells[1].text = s
        row.cells[2].text = v
        row.cells[3].text = req_val
        row.cells[4].text = st_val

    # Soil table
    doc.add_heading("Soil & Water Parameters", level=2)
    t_soil = doc.add_table(rows=1, cols=3)
    t_soil.style = 'Table Grid'
    hdr = t_soil.rows[0].cells
    hdr[0].text = 'Parameter'
    hdr[1].text = 'Symbol'
    hdr[2].text = 'Value'
    
    soil_items = [
        ("Dry unit weight", "γ_dry", f"{gamma_soil_dry:.2f} kN/m³"),
        ("Wet unit weight", "γ_wet", f"{gamma_soil_wet:.2f} kN/m³"),
        ("Friction angle", "φ", f"{phi:.1f}°"),
        ("Soil cohesion", "c", f"{c_soil:.1f} kPa"),
        ("Surcharge load", "q", f"{q:.2f} kPa"),
        ("Soil height above heel", "h_soil", f"{h_soil:.3f} m"),
        ("Soil above toe", "h_toe", f"{h_soil_toe:.3f} m"),
        ("Water unit weight", "γ_w", f"{gamma_w:.2f} kN/m³"),
        ("Water height behind wall", "Hwtr", f"{Hwtr:.2f} m"),
        ("Water height in front", "Hwtr_front", f"{Hwtr_front:.3f} m")
    ]
    for p, s, v in soil_items:
        row = t_soil.add_row()
        row.cells[0].text = p
        row.cells[1].text = s
        row.cells[2].text = v
        
    # 2. Geometry diagram
    doc.add_heading("2. Model & Geometry Diagrams", level=1)
    if os.path.exists("report_temp/model_geometry.png"):
        doc.add_heading("Wall Profile & Mesh Grid", level=2)
        doc.add_picture("report_temp/model_geometry.png", width=Inches(4.5))
    if os.path.exists("report_temp/typical_dimensions.png"):
        doc.add_heading("Dimensions Proportions Verification", level=2)
        doc.add_picture("report_temp/typical_dimensions.png", width=Inches(4.5))
        
    # 3. Applied Loads
    doc.add_heading("3. Applied Loads & Pressure Distributions", level=1)
    
    h_active = min(Hw, h_soil)
    if surcharge_type == 'Strip Load':
        q_top = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - h_active))['delta sigma x [kPa]'] if h_soil > 0 else 0.0
        q_bot = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
    else:
        q_top = Ka * q if h_soil > 0 else 0.0
        q_bot = Ka * q if h_soil > 0 else 0.0
    p_top_act = Ka * (gamma_soil_dry * max(0.0, h_soil - h_active)) + q_top if h_soil > 0 else 0.0
    p_bot_act = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot if h_soil > 0 else 0.0
    
    FS_passive_val = 2.0
    Kp_val = np.tan(np.radians(45.0 + phi / 2.0)) ** 2
    p_p_top = (Kp_val * gamma_soil_dry * h_soil_toe) / FS_passive_val if h_soil_toe > 0 else 0.0
    p_p_bot = (Kp_val * gamma_soil_dry * (h_soil_toe + h_ftg)) / FS_passive_val
    
    p_v = q_soil + q
    p_w_back = gamma_w * Hwtr
    p_w_front = gamma_w * Hwtr_front
    p_upl = gamma_w * Hwtr

    # Soil Lateral Pressure
    if os.path.exists("report_temp/load_soil_lateral.png"):
        doc.add_heading("A. Soil Lateral Pressure (Active & Passive)", level=2)
        doc.add_picture("report_temp/load_soil_lateral.png", width=Inches(4.5))
        
        doc.add_paragraph("Active Earth Coefficient Equation:")
        if os.path.exists("report_temp/eq_ka.png"):
            doc.add_picture("report_temp/eq_ka.png", width=Inches(2.0))
        doc.add_paragraph(f"Result: Ka = {Ka:.4f}")
        
        doc.add_paragraph("Active Pressure (Top of Stem):")
        if os.path.exists("report_temp/eq_p_act_top.png"):
            doc.add_picture("report_temp/eq_p_act_top.png", width=Inches(1.8))
        doc.add_paragraph(f"Result: p_act,top = {p_top_act:.2f} kPa")
        
        doc.add_paragraph("Active Pressure (Bottom of Footing):")
        if os.path.exists("report_temp/eq_p_act_bot.png"):
            doc.add_picture("report_temp/eq_p_act_bot.png", width=Inches(2.8))
        doc.add_paragraph(f"Result: p_act,bot = {p_bot_act:.2f} kPa")
        
        doc.add_paragraph("Passive Earth Coefficient Equation:")
        if os.path.exists("report_temp/eq_kp.png"):
            doc.add_picture("report_temp/eq_kp.png", width=Inches(2.0))
        doc.add_paragraph(f"Result: Kp = {Kp_val:.4f}")
        
        doc.add_paragraph("Passive Pressure (Bottom of Footing):")
        if os.path.exists("report_temp/eq_p_pass_bot.png"):
            doc.add_picture("report_temp/eq_p_pass_bot.png", width=Inches(2.4))
        doc.add_paragraph(f"Result: p_pass,bot = {p_p_bot:.2f} kPa (Mobilized, FS_pass = 2.0)")

    # Soil Vertical Weight
    if os.path.exists("report_temp/load_soil_vertical.png"):
        doc.add_heading("B. Soil Vertical Weight & Surcharge", level=2)
        doc.add_picture("report_temp/load_soil_vertical.png", width=Inches(4.5))
        doc.add_paragraph("Vertical Backfill Pressure Equation:")
        if os.path.exists("report_temp/eq_p_v_heel.png"):
            doc.add_picture("report_temp/eq_p_v_heel.png", width=Inches(1.8))
        doc.add_paragraph(f"Result: p_v,heel = {p_v:.2f} kPa")

    # Hydrostatic Water
    if os.path.exists("report_temp/load_hydrostatic.png"):
        doc.add_heading("C. Hydrostatic Water Pressure", level=2)
        doc.add_picture("report_temp/load_hydrostatic.png", width=Inches(4.5))
        doc.add_paragraph("Hydrostatic Water Pressure Equation:")
        if os.path.exists("report_temp/eq_p_w.png"):
            doc.add_picture("report_temp/eq_p_w.png", width=Inches(1.5))
        doc.add_paragraph(f"Result: Backwater p_w,back = {p_w_back:.2f} kPa, Frontwater p_w,front = {p_w_front:.2f} kPa")

    # Uplift
    if os.path.exists("report_temp/load_uplift.png"):
        doc.add_heading("D. Uplift Water Pressure under Base", level=2)
        doc.add_picture("report_temp/load_uplift.png", width=Inches(4.5))
        doc.add_paragraph("Base Uplift Pressure Equation:")
        if os.path.exists("report_temp/eq_p_uplift.png"):
            doc.add_picture("report_temp/eq_p_uplift.png", width=Inches(1.8))
        doc.add_paragraph(f"Result: p_uplift = {p_upl:.2f} kPa")

    # Seismic
    if os.path.exists("report_temp/load_seismic.png"):
        doc.add_heading("E. Pseudo-Static Seismic Forces", level=2)
        doc.add_picture("report_temp/load_seismic.png", width=Inches(4.5))

    # Total
    if os.path.exists("report_temp/load_total.png"):
        doc.add_heading("F. Total / Combined Load System", level=2)
        doc.add_picture("report_temp/load_total.png", width=Inches(4.5))

    # 4. Stability
    doc.add_heading("4. Stability Analysis & Verification", level=1)
    doc.add_heading("Stability Verification Equations", level=2)
    doc.add_paragraph("Sliding safety factor equation:")
    if os.path.exists("report_temp/eq_sliding.png"):
         doc.add_picture("report_temp/eq_sliding.png", width=Inches(2.8))
    doc.add_paragraph("Overturning safety factor equation:")
    if os.path.exists("report_temp/eq_overturning.png"):
         doc.add_picture("report_temp/eq_overturning.png", width=Inches(2.4))

    if 'FS_slide_global' in globals() and FS_slide_global is not None:
        doc.add_paragraph(f"Sliding FS = {FS_slide_global:.3f} (Required >= 1.5)")
    if 'FS_ot_global' in globals() and FS_ot_global is not None:
        doc.add_paragraph(f"Overturning FS = {FS_ot_global:.3f} (Required >= 1.5)")
    if 'fs_ssrm' in globals() and fs_ssrm is not None:
        doc.add_paragraph(f"Global Slope Stability SSRM FS = {fs_ssrm:.3f}")
        
    if os.path.exists("report_temp/xslope_inputs.png"):
        doc.add_heading("XSLOPE Slope Geometry & Inputs Diagram", level=2)
        doc.add_picture("report_temp/xslope_inputs.png", width=Inches(4.5))
    if os.path.exists("report_temp/xslope_mesh.png"):
        doc.add_heading("XSLOPE FEM Mesh with Material Zones (tri6)", level=2)
        doc.add_picture("report_temp/xslope_mesh.png", width=Inches(4.5))
    if os.path.exists("report_temp/stability_ssrm.png"):
        doc.add_heading("2D SSRM Viscoplastic Shear Strain Failure Surface", level=2)
        doc.add_picture("report_temp/stability_ssrm.png", width=Inches(4.5))
        
    # 5. Bearing Capacity
    if enable_bearing:
        doc.add_heading("5. Bearing Capacity & Settlement", level=1)
        doc.add_heading("Bearing Stress & Settlement Equations", level=2)
        doc.add_paragraph("Bearing contact stress:")
        if os.path.exists("report_temp/eq_bearing_stress.png"):
             doc.add_picture("report_temp/eq_bearing_stress.png", width=Inches(2.2))
        doc.add_paragraph("Bearing safety factor:")
        if os.path.exists("report_temp/eq_bearing_fs.png"):
             doc.add_picture("report_temp/eq_bearing_fs.png", width=Inches(1.8))
        doc.add_paragraph("Consolidation settlement:")
        if os.path.exists("report_temp/eq_settlement.png"):
             doc.add_picture("report_temp/eq_settlement.png", width=Inches(2.5))

        if 'q_ult' in globals() and q_ult is not None:
            doc.add_paragraph(f"Ultimate bearing capacity: {q_ult:.2f} kPa")
            doc.add_paragraph(f"Max bearing stress: {sigma_max:.2f} kPa")
            doc.add_paragraph(f"Safety factor: {q_ult/sigma_max:.2f} (Required >= {FS_bearing:.1f})")
        if 'consol_settlement_mm' in globals() and consol_settlement_mm is not None:
            doc.add_paragraph(f"Consolidation Settlement: {consol_settlement_mm:.2f} mm (Allowable: {consol_allow_mm:.2f} mm)")

    # 6. FEM
    doc.add_heading("6. Finite Element Settlement & Base Displacements", level=1)
    doc.add_paragraph(f"Toe Settlement: {settlement_toe_mm:.2f} mm")
    doc.add_paragraph(f"Heel Settlement: {settlement_heel_mm:.2f} mm")
    doc.add_paragraph(f"Differential Settlement: {diff_settlement_mm:.2f} mm (Allowable: 50.00 mm)")
    
    # 7. Stress Contours
    doc.add_heading("7. Finite Element Stress & Bending Moment Contours", level=1)
    if os.path.exists("report_temp/stem_bending_moment.png"):
        doc.add_heading("Stem Node Bending Moment Diagram (kN·m)", level=2)
        doc.add_picture("report_temp/stem_bending_moment.png", width=Inches(4.5))
    if os.path.exists("report_temp/stress_xx.png"):
        doc.add_heading("FE Stress Contour σ_xx (kN/m²)", level=2)
        doc.add_picture("report_temp/stress_xx.png", width=Inches(4.5))
    if os.path.exists("report_temp/stress_yy.png"):
        doc.add_heading("FE Stress Contour σ_yy (kN/m²)", level=2)
        doc.add_picture("report_temp/stress_yy.png", width=Inches(4.5))

    # 8. Pile Foundation
    if enable_pile:
        doc.add_heading("8. Pile Foundation Design & Verification", level=1)
        doc.add_heading("Pile Cross Section Geometry & Material Details", level=2)
        
        t_pile = doc.add_table(rows=1, cols=3)
        t_pile.style = 'Table Grid'
        hdr = t_pile.rows[0].cells
        hdr[0].text = 'Parameter Name'
        hdr[1].text = 'Symbol'
        hdr[2].text = 'Value'
        
        pile_param_data = [
            ["Pile Material Type", "-", pile_material],
            ["Pile Cross-section Shape", "-", pile_shape],
        ]
        if pile_material == 'Concrete':
            if pile_shape == 'Circle':
                pile_param_data.append(["Pile Diameter", "d_pile", f"{diameter_pile:.3f} m"])
                pile_param_data.append(["Concrete Cover", "d_c", f"{cover_pile:.1f} mm"])
                pile_param_data.append(["Main Reinforcement", "-", f"{int(n_rebar_pile)} D{rebar_dia_pile}"])
            else:
                pile_param_data.append(["Pile Dimensions (dx × dy)", "h_pile × b_pile", f"{width_x_pile:.3f} m × {width_y_pile:.3f} m"])
                pile_param_data.append(["Concrete Cover", "d_c", f"{cover_pile:.1f} mm"])
                pile_param_data.append(["Main Reinforcement", "-", f"4 D{rebar_dia_pile} (Corners)"])
        else:
            if pile_shape == 'Circle':
                pile_param_data.append(["Pile Diameter", "d_pile", f"{diameter_pile:.3f} m"])
            else:
                pile_param_data.append(["Pile Dimensions (dx × dy)", "h_pile × b_pile", f"{width_x_pile:.3f} m × {width_y_pile:.3f} m"])
                
        for p, s, v in pile_param_data:
            row = t_pile.add_row()
            row.cells[0].text = p
            row.cells[1].text = s
            row.cells[2].text = v

        if os.path.exists("report_temp/pile_section.png"):
            doc.add_heading("Pile Cross-Section Drawing", level=2)
            doc.add_picture("report_temp/pile_section.png", width=Inches(3.5))

        doc.add_heading("Pile Demands & Safety Factors Checklist", level=2)
        if 'P_toe' in globals() and P_toe is not None:
            doc.add_paragraph(f"Toe Pile Axial demand: {P_toe:.2f} kN")
            doc.add_paragraph(f"Toe Pile Lateral demand: {V_max_toe:.2f} kN")
        if 'P_heel' in globals() and P_heel is not None:
            doc.add_paragraph(f"Heel Pile Axial demand: {P_heel:.2f} kN")
            doc.add_paragraph(f"Heel Pile Lateral demand: {V_max_heel:.2f} kN")
            
        if os.path.exists("report_temp/pile_elevation.png"):
            doc.add_heading("Pile Foundation Elevation Schematic", level=2)
            doc.add_picture("report_temp/pile_elevation.png", width=Inches(4.5))
        if os.path.exists("report_temp/pile_winkler.png"):
            doc.add_heading("Pile Winkler Analysis Elastic Profiles", level=2)
            doc.add_picture("report_temp/pile_winkler.png", width=Inches(4.5))
        if os.path.exists("report_temp/pile_interaction.png"):
            doc.add_heading("Moment-Axial Interaction Capacity Curve", level=2)
            doc.add_picture("report_temp/pile_interaction.png", width=Inches(4.5))
            
    doc.save(output_path)

@st.dialog("RT Wall Cantilever Report Export", width="large")
def show_report_dialog():
    st.markdown("### 📋 RT Wall Cantilever Design Report Export")
    st.markdown("Generate and download a comprehensive engineering design report for the current wall configuration in Word (**DOCX**) or **PDF** format.")

    with st.spinner("Preparing report assets and figures..."):
        export_report_assets()
        
    pdf_filename = "retaining_wall_report.pdf"
    docx_filename = "retaining_wall_report.docx"
    
    pdf_path = os.path.join("report_temp", pdf_filename)
    docx_path = os.path.join("report_temp", docx_filename)
    
    col_docx, col_pdf = st.columns(2)
    
    with col_docx:
        try:
            generate_docx_report(docx_path)
            with open(docx_path, "rb") as f:
                docx_bytes = f.read()
            st.download_button(
                label="📝 Download Word Document (.docx)",
                data=docx_bytes,
                file_name=docx_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        except Exception as e_docx:
            st.error(f"Error generating Word report: {e_docx}")
            
    with col_pdf:
        try:
            generate_pdf_reportlab(pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📕 Download PDF Report (.pdf)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e_pdf:
            st.error(f"Error generating PDF: {e_pdf}")
            
    st.markdown("---")
    
    # Safely extract preview values
    slide_val = FS_slide_global if ('FS_slide_global' in globals() and FS_slide_global is not None) else 0.0
    ot_val = FS_ot_global if ('FS_ot_global' in globals() and FS_ot_global is not None) else 0.0
    slide_pass = "PASS" if slide_val >= 1.5 else "FAIL"
    ot_pass = "PASS" if ot_val >= 1.5 else "FAIL"

    st.markdown("#### 🔍 Design Summary Preview")
    st.markdown(f"""
    - **Wall Height**: {Hw:.3f} m | **Footing Width**: {ftg:.3f} m
    - **Backfill Soil Friction Angle**: {phi:.1f}° | **Cohesion**: {c_soil:.1f} kPa
    - **Sliding safety check**: **{slide_pass}** (FS = {slide_val:.3f})
    - **Overturning safety check**: **{ot_pass}** (FS = {ot_val:.3f})
    """)
    
    # Real-time base64 PDF Preview frame
    try:
        import base64
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown("#### 📄 PDF Inline Preview")
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e_prev:
        st.warning(f"Unable to load PDF preview frame: {e_prev}")
# Page Configuration for wide layout
st.set_page_config(layout="wide", page_title="RT Wall Cantilever", page_icon="🧱")


# CSS Injection for Premium Styling
st.markdown("""
<style>
    /* Hide Streamlit default status widget / top running banner */
    [data-testid="stStatusWidget"],
    div[data-testid="stStatusWidget"],
    .stStatusWidget,
    div[class*="stStatusWidget"],
    header [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
    }
    /* Gradient Dashboard Header Banner */
    .dashboard-header {
        background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
        color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 24px;
        text-align: center;
    }
    .dashboard-header h1 {
        margin: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        margin: 8px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    /* Section subtitle details */
    .section-desc {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

def update_splash(placeholder, pct, title_text, desc_text):
    if '_qt_progress_hook' in globals() and callable(_qt_progress_hook):
        _qt_progress_hook(pct, title_text, desc_text)
    if placeholder is None:
        return
    circ = 276.46
    offset = circ * (1.0 - float(pct) / 100.0)
    splash_html = f"""
    <style>
    [data-testid="stStatusWidget"],
    div[data-testid="stStatusWidget"],
    .stStatusWidget,
    div[class*="stStatusWidget"],
    header [data-testid="stStatusWidget"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
    }}
    .splash-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        z-index: 999999;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .splash-card {{
        background: #ffffff;
        border-radius: 20px;
        padding: 40px 48px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        text-align: center;
        max-width: 480px;
        width: 90%;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }}
    .circular-progress {{
        width: 110px;
        height: 110px;
        margin: 0 auto 24px auto;
        position: relative;
    }}
    .circular-progress svg {{
        width: 100%;
        height: 100%;
        transform: rotate(-90deg);
    }}
    .circular-progress circle {{
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
    }}
    .circle-bg {{
        stroke: #e2e8f0;
    }}
    .circle-fill {{
        stroke: #1f4068;
        transition: stroke-dashoffset 0.3s ease-in-out;
    }}
    .progress-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.45rem;
        font-weight: 700;
        color: #162447;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .splash-title {{
        margin: 0 0 8px 0;
        color: #162447;
        font-size: 1.25rem;
        font-weight: 700;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .splash-desc {{
        margin: 0;
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.45;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    </style>
    <div class="splash-overlay">
        <div class="splash-card">
            <div class="circular-progress">
                <svg viewBox="0 0 100 100">
                    <circle class="circle-bg" cx="50" cy="50" r="44"></circle>
                    <circle class="circle-fill" cx="50" cy="50" r="44" stroke-dasharray="276.46" stroke-dashoffset="{offset:.2f}"></circle>
                </svg>
                <div class="progress-text">{pct}%</div>
            </div>
            <div class="splash-title">{title_text}</div>
            <div class="splash-desc">{desc_text}</div>
        </div>
    </div>
    """
    placeholder.markdown(splash_html, unsafe_allow_html=True)

# ----------------------------------------

# Clean up anything created by invoked opensees commands
ops.wipe()

# We define the space in which we will create the model
ops.model('basic','-ndm',2,'-ndf',2) # Displacements in 2 directions, out-of-plane displacements and rotations are restricted.

# Sidebar header
st.sidebar.markdown("### ⚙️ Input Configuration")

# Streamlit sidebar inputs (re-runs app on change)
tab_geom, tab_soil, tab_struct, tab_seismic, tab_reinf, tab_bearing, tab_pile = st.sidebar.tabs(['Geometry', 'Soil', 'Structure', 'Seismic', 'Concrete & Reinf', 'Bearing Capacity', 'Pile'])

with tab_geom:
    t = st.number_input('Out-of-plane thickness (m)', value=1.0, step=0.01, format="%.3f")
    Hw = st.number_input('Wall height (m)', value=3.0, step=0.01, format="%.3f")
    toe = st.number_input('Toe length (m)', value=0.8, step=0.01, format="%.3f")
    heel = st.number_input('Heel length (m)', value=0.8, step=0.01, format="%.3f")
    top_wall = st.number_input('Top wall thickness (m)', value=0.3, step=0.01, format="%.3f")
    bot_wall = st.number_input('Bottom wall thickness (m)', value=0.4, step=0.01, format="%.3f")
    taper_direction = st.selectbox('Taper Direction', ['Toe-facing', 'Heel-facing'], index=0, help="Toe-facing tapers the left face of the stem; Heel-facing tapers the right (backfill) face of the stem.")
    h_ftg = st.number_input('Footing thickness (m)', value=0.3, step=0.01, format="%.3f")
    # Shear key optional parameters
    include_shear_key = st.checkbox('Include Shear Key', value=False)
    if include_shear_key:
        st.subheader('Shear Key Parameters')
        shear_key_distance = st.number_input('Distance from toe edge (m)', value=0.1, step=0.01, format="%.3f")
        shear_key_width = st.number_input('Shear key width (m)', value=0.5, step=0.01, format="%.3f")
        shear_key_thickness = st.number_input('Shear key thickness (m)', value=0.2, step=0.01, format="%.3f")

with tab_soil:
    # Soil parameters
    gamma_soil_dry = st.number_input('Dry soil unit weight (kN/m3)', value=18.0, step=0.1, format="%.2f")
    gamma_soil_wet = st.number_input('Wet soil unit weight (kN/m3)', value=20.0, step=0.1, format="%.2f")
    phi = st.number_input('Friction angle (deg)', value=45, step=1)
    surcharge_type = st.selectbox('Surcharge Type', ['Uniform', 'Strip Load'], index=0)
    q = st.number_input('Surcharge load q (kN/m2)', value=10.0, step=0.1, format="%.2f")
    if surcharge_type == 'Strip Load':
        width_surcharge = st.number_input('Surcharge width B (m)', value=2.0, step=0.1, format="%.2f")
        offset_surcharge = st.number_input('Surcharge offset from wall face a (m)', value=1.5, step=0.1, format="%.2f")
    else:
        width_surcharge = 1e5
        offset_surcharge = 0.0
    h_soil = st.number_input('Soil height above heel (m)', value=3.0, step=0.01, format="%.3f")
    h_soil_toe = st.number_input('Soil above toe (m)', value=1.0, step=0.01, format="%.3f")
    c_soil = st.number_input('Soil cohesion for slope stability (kPa)', value=10.0, step=1.0, format="%.1f")

    # Water parameters
    gamma_w = st.number_input('Water unit weight (kN/m3)', value=9.81, step=0.01, format="%.2f")
    Hwtr = st.number_input('Water height behind wall (m)', value=0.5, step=0.01, format="%.2f")
    Hwtr_front = st.number_input('Water height in front of wall (m)', value=0.15, step=0.01, format="%.3f")

    # Soil Spring Parameters (for Winkler subgrade model)
    st.subheader("Soil Spring Properties")
    Es_soil = st.number_input('Soil Young\'s Modulus Es (kPa)', value=40000.0, step=1000.0, format="%.1f", help="Typical: Sand 10,000-50,000 kPa; Clay 5,000-25,000 kPa")
    nu_soil = st.number_input('Soil Poisson\'s Ratio ν', value=0.15, step=0.05, format="%.2f", min_value=0.0, max_value=0.49)

with tab_seismic:
    #Seismic Parameter
    PGA = st.number_input('Peak Ground Acceleration (g)', value=0.4123, step=0.01, format="%.2f")
    FPGA = st.number_input('Seismic Load Factor', value=1.2, step=0.01, format="%.2f")

with tab_struct:
    # Concrete self-weight
    gamma_c = st.number_input('Unit weight of concrete (kN/m3)', value=24.0, step=0.1, format="%.2f")
    c_concrete = st.number_input('Concrete cohesion (kPa)', value=180.0, step=10.0, format="%.1f")


with tab_reinf:
    st.subheader("Concrete Properties")
    fc = st.number_input('Comp. strength f\'c (MPa)', value=50.0, step=5.0)
    Ec = st.number_input('Elastic Modulus Ec (MPa)', value=34.8e3, step=1000.0)
    ft_tensile = st.number_input('Flexural tensile strength (MPa)', value=4.2, step=0.1)
    
    st.subheader("Steel Properties")
    fy = st.number_input('Yield strength fsy (MPa)', value=500.0, step=50.0)
    Es = st.number_input('Elastic Modulus Es (MPa)', value=200e3, step=5000.0)
    
    # Indonesian Standard SNI Rebar diameter lists in mm
    sni_diameters = [6, 8, 10, 12, 13, 14, 16, 19, 22, 25, 29, 32, 36, 40, 50]
    
    st.subheader("Stem Reinforcement")
    stem_rebar_dia = st.selectbox('Stem main bar diameter (mm)', sni_diameters, index=9, help="SNI diameter options (e.g. 25 = D25)")
    stem_bar_area = np.pi * (stem_rebar_dia ** 2) / 4.0
    st.caption(f"Calculated Stem Main Bar Area: **{stem_bar_area:.2f} mm²**")
    
    stem_spacing_x = st.number_input('Stem bar spacing x (mm)', value=150.0, step=10.0)
    
    stem_sidebar_dia = st.selectbox('Stem side bar diameter (mm)', sni_diameters, index=6)
    stem_sidebar_area = np.pi * (stem_sidebar_dia ** 2) / 4.0
    st.caption(f"Calculated Stem Side Bar Area: **{stem_sidebar_area:.2f} mm²**")
    
    stem_spacing_y = st.number_input('Stem side bar spacing y (mm)', value=100.0, step=10.0)
    stem_cover = st.number_input('Stem concrete cover (mm)', value=50.0, step=5.0)
    
    st.subheader("Footing Reinforcement")
    ftg_rebar_dia = st.selectbox('Footing main bar diameter (mm)', sni_diameters, index=9)
    ftg_bar_area = np.pi * (ftg_rebar_dia ** 2) / 4.0
    st.caption(f"Calculated Footing Main Bar Area: **{ftg_bar_area:.2f} mm²**")
    
    ftg_spacing_x = st.number_input('Footing bar spacing x (mm)', value=150.0, step=10.0)
    
    ftg_sidebar_dia = st.selectbox('Footing side bar diameter (mm)', sni_diameters, index=6)
    ftg_sidebar_area = np.pi * (ftg_sidebar_dia ** 2) / 4.0
    st.caption(f"Calculated Footing Side Bar Area: **{ftg_sidebar_area:.2f} mm²**")
    
    ftg_spacing_y = st.number_input('Footing side bar spacing y (mm)', value=200.0, step=10.0)
    ftg_cover = st.number_input('Footing concrete cover (mm)', value=50.0, step=5.0)

with tab_bearing:
    enable_bearing = st.checkbox('Enable Bearing Capacity Analysis', value=False)
    if enable_bearing:
        bearing_soil_type = st.selectbox('Soil Type for Bearing', ['Sand (Drained)', 'Clay (Undrained)'], index=0)
        if bearing_soil_type == 'Clay (Undrained)':
            su_val = st.number_input('Undrained shear strength su (kPa)', value=50.0, step=5.0, format="%.1f")
        else:
            su_val = 0.0
        FS_bearing = st.number_input('Bearing Capacity Safety Factor (FS)', value=3.0, step=0.1, format="%.2f")
        
        st.subheader("Consolidation Settlement")
        consol_type = st.selectbox('Consolidation Type', ['Normally Consolidated (NC)', 'Overconsolidated (OC)'], index=0)
        H0_consol = st.number_input('Compressible Layer Thickness H₀ (m)', value=1.0, step=0.5, format="%.1f")
        e0_consol = st.number_input('Initial Void Ratio e₀', value=0.8, step=0.05, format="%.2f")
        Cc_consol = st.number_input('Compression Index Cc', value=0.3, step=0.05, format="%.3f")
        if consol_type == 'Overconsolidated (OC)':
            Cr_consol = st.number_input('Recompression Index Cr', value=0.05, step=0.01, format="%.3f")
            pc_consol = st.number_input('Preconsolidation Pressure p\'c (kPa)', value=200.0, step=10.0, format="%.1f")
        else:
            Cr_consol = 0.05
            pc_consol = 200.0
    else:
        bearing_soil_type = 'Sand (Drained)'
        su_val = 0.0
        consol_type = 'Normally Consolidated (NC)'
        H0_consol = 5.0
        e0_consol = 0.8
        Cc_consol = 0.3
        Cr_consol = 0.05
        pc_consol = 200.0
        
with tab_pile:
    enable_pile = st.checkbox('Enable Pile Foundation Analysis', value=False)
    if enable_pile:
        pile_shape = st.selectbox('Pile Shape', ['Circle', 'Rectangle'], index=0)
        pile_material = st.selectbox('Pile Material', ['Concrete', 'Steel'], index=0)
        pile_offset = st.number_input('Offset from Edge to Pile Centroid (m)', value=0.50, step=0.05, format="%.2f")
        L_pile = st.number_input('Pile Length (m)', value=10.0, step=0.5, format="%.1f")
        
        if pile_shape == 'Circle':
            diameter_pile = st.number_input('Pile Diameter (m)', value=0.40, step=0.05, format="%.2f")
            width_x_pile = diameter_pile
            width_y_pile = diameter_pile
        else:
            width_x_pile = st.number_input('Pile Width in Bending direction dx (m)', value=0.40, step=0.05, format="%.2f")
            width_y_pile = st.number_input('Pile Width perpendicular to Bending dy (m)', value=0.40, step=0.05, format="%.2f")
            diameter_pile = width_x_pile
            
        if pile_material == 'Concrete':
            n_rebar_pile = st.number_input('Number of Pile Rebars', value=8, min_value=4, step=1)
            rebar_dia_pile = st.selectbox('Pile Rebar Diameter (mm)', sni_diameters, index=9)
            cover_pile = st.number_input('Pile Concrete Cover (mm)', value=50.0, step=5.0)
        else:
            n_rebar_pile = 8
            rebar_dia_pile = 25
            cover_pile = 50.0
        
        st.subheader("Soil Profile under Footing")
        pile_soil_type = st.selectbox('Soil Type for Piles', ['Sand (Drained)', 'Clay (Undrained)'], index=0)
        if pile_soil_type == 'Sand (Drained)':
            phi_pile_soil = st.number_input('Pile Soil Friction Angle (deg)', value=30.0, step=1.0, format="%.1f")
            cohesion_pile_soil = st.number_input('Pile Soil Cohesion (kPa)', value=0.0, step=1.0, format="%.1f")
            gamma_pile_soil = st.number_input('Pile Soil Unit Weight (kN/m3)', value=18.0, step=0.5, format="%.1f")
            su_pile_soil = 0.0
            eps50_pile_soil = 0.015
        else:
            su_pile_soil = st.number_input('Pile Soil Undrained Shear Strength su (kPa)', value=50.0, step=5.0, format="%.1f")
            gamma_pile_soil = st.number_input('Pile Soil Unit Weight (kN/m3)', value=18.0, step=0.5, format="%.1f")
            eps50_pile_soil = st.number_input('Pile Soil Strain at 50% max stress eps50', value=0.015, step=0.005, format="%.3f")
            phi_pile_soil = 30.0
            cohesion_pile_soil = 0.0
            
        st.subheader("Analysis Safety Factors")
        FS_pile_axial = st.number_input('Axial Safety Factor (piles)', value=2.5, step=0.1, format="%.2f")
        FS_pile_lateral = st.number_input('Lateral Safety Factor (piles)', value=2.5, step=0.1, format="%.2f")
        pile_loading_type = st.selectbox('OpenPile Loading Type', ['static', 'cyclic'], index=0)
        
        st.subheader("Pile Axial Settlement")
        if pile_soil_type == 'Sand (Drained)':
            delta_pile_axial = st.number_input('Pile-Soil Interface Friction Angle δ (deg)', value=round(phi_pile_soil * 0.7, 1), step=1.0, format="%.1f")
            alpha_pile_axial = 0.5
        else:
            alpha_pile_axial = st.number_input('Adhesion Factor α (clay)', value=0.5, step=0.05, format="%.2f")
            delta_pile_axial = 20.0
    else:
        pile_shape = 'Circle'
        pile_material = 'Concrete'
        pile_offset = 0.50
        L_pile = 10.0
        diameter_pile = 0.40
        width_x_pile = 0.40
        width_y_pile = 0.40
        n_rebar_pile = 8
        rebar_dia_pile = 25
        cover_pile = 50.0
        pile_soil_type = 'Sand (Drained)'
        phi_pile_soil = 30.0
        cohesion_pile_soil = 0.0
        su_pile_soil = 50.0
        gamma_pile_soil = 18.0
        eps50_pile_soil = 0.015
        FS_pile_axial = 2.5
        FS_pile_lateral = 2.5
        pile_loading_type = 'static'
# Sidebar Controls: Run Analysis & Generate Report
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Analysis & Report Controls")

run_analysis_clicked = st.sidebar.button("🚀 Run Analysis", key="btn_run_analysis", type="primary", use_container_width=True)

if run_analysis_clicked:
    st.session_state["analysis_run"] = True
    st.session_state["trigger_splash"] = True

has_analysis_run = st.session_state.get("analysis_run", False)

if has_analysis_run:
    if st.sidebar.button("📄 Generate Report", key="btn_report_sidebar", use_container_width=True):
        st.session_state["trigger_report_splash"] = True
else:
    st.sidebar.info("💡 Click **'🚀 Run Analysis'** above to run stability analysis & enable report export.")

splash_placeholder = None
if st.session_state.get("trigger_splash", False):
    splash_placeholder = st.empty()
    update_splash(
        splash_placeholder,
        15,
        "Validating Geometry & Soil Profile",
        "Checking stem wall dimensions, soil parameters, and surcharge loadings..."
    )

# Derived geometry
ftg = toe + heel + bot_wall  # total footing length
taper = bot_wall - top_wall
taper_length = np.sqrt(taper**2 + Hw**2)

# Taper-dependent coordinate bounds
top_left_stem_x = toe + taper if taper_direction == 'Toe-facing' else toe
top_right_stem_x = toe + bot_wall if taper_direction == 'Toe-facing' else toe + top_wall
batter_mid_x = toe + taper/2.0 if taper_direction == 'Toe-facing' else toe + bot_wall - taper/2.0

# Generic stem points for SVG drawings (elevation coordinates relative to footing top at y=0)
if taper_direction == 'Toe-facing':
    stem_pts = [
        ((toe + taper/4.0), -(h_ftg + Hw/4.0)),
        ((toe + taper/2.0), -(h_ftg + Hw/2.0)),
        ((toe + taper*3.0/4.0), -(h_ftg + Hw*3.0/4.0)),
        ((toe + taper), -(h_ftg + Hw)),
        ((toe + bot_wall), -(h_ftg + Hw)),
        ((toe + bot_wall), -(h_ftg + Hw*3.0/4.0)),
        ((toe + bot_wall), -(h_ftg + Hw/2.0)),
        ((toe + bot_wall), -(h_ftg + Hw/4.0)),
        ((toe + bot_wall), -h_ftg)
    ]
else:
    stem_pts = [
        (toe, -(h_ftg + Hw/4.0)),
        (toe, -(h_ftg + Hw/2.0)),
        (toe, -(h_ftg + Hw*3.0/4.0)),
        (toe, -(h_ftg + Hw)),
        ((toe + top_wall), -(h_ftg + Hw)),
        ((toe + bot_wall - taper*3.0/4.0), -(h_ftg + Hw*3.0/4.0)),
        ((toe + bot_wall - taper/2.0), -(h_ftg + Hw/2.0)),
        ((toe + bot_wall), -h_ftg)
    ]

if not has_analysis_run:
    st.markdown("""
    <div class="dashboard-header">
        <h1>🧱 RT Wall Cantilever</h1>
        <p>Cantilever Retaining Wall Design & Verification Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("### 👈 Ready to Analyze")
        st.markdown("""
        Configure your retaining wall dimensions, soil parameters, structural reinforcement, and pile foundation options in the sidebar tabs.
        
        Once ready, click **'🚀 Run Analysis'** in the sidebar to execute stability analysis, render the design dashboard, and enable report export.
        """)
    st.stop()

# Build a refined mesh by splitting the original triangles while keeping
# the overall outer geometry unchanged. We create 8 nodes along the
# footing (bottom and top of footing) and 8 stem layers (back/front
# pairs) resulting in 32 nodes and 30 Tri31 elements.

# compute x positions along footing by subdividing the original three
# segments (0->toe, toe->toe+bot_wall, toe+bot_wall->ftg) into 3,2,2
# subdivisions respectively so total nodes along footing = 1+3+2+2 = 8
seg1_div = 3
seg2_div = 2
seg3_div = 2

xs = []
# Segment 1: 0 -> toe
for i in range(seg1_div):
    xs.append(0.0 + (toe - 0.0) * (i / float(seg1_div)))
xs.append(float(toe))
# Segment 2: toe -> toe+bot_wall
for i in range(1, seg2_div + 1):
    xs.append(float(toe) + (bot_wall) * (i / float(seg2_div)))
# Segment 3: toe+bot_wall -> ftg
for i in range(1, seg3_div + 1):
    xs.append((toe + bot_wall) + (ftg - (toe + bot_wall)) * (i / float(seg3_div)))

# ensure unique and sorted (floating tolerance)
xs_unique = []
for x in xs:
    if not any(abs(x - xx) < 1e-9 for xx in xs_unique):
        xs_unique.append(x)

# bottom nodes (y=0): IDs 1..8
num_foot_nodes = len(xs_unique)
bottom_node_ids = list(range(1, num_foot_nodes + 1))
for idx, x in enumerate(xs_unique):
    nid = bottom_node_ids[idx]
    ops.node(nid, float(x), 0.0)

# top-of-footing nodes (y=h_ftg): IDs 9..16
top_node_ids = list(range(num_foot_nodes + 1, num_foot_nodes * 2 + 1))
for idx, x in enumerate(xs_unique):
    nid = top_node_ids[idx]
    ops.node(nid, float(x), h_ftg)

# Stem nodes: create 8 back/front pairs above the top-of-footing
# back nodes follow the tapered profile, front nodes at x = toe+bot_wall
# IDs 17..32 (back1,front1, back2,front2, ...)
stem_pairs = []
shear_node_ids = []  # will hold shear key node ids if enabled
for i in range(1, 9):
    z = h_ftg + Hw * (i / 8.0)
    if taper_direction == 'Toe-facing':
        back_x = toe + taper * (i / 8.0)
        front_x = toe + bot_wall
    else:
        back_x = toe
        front_x = toe + bot_wall - taper * (i / 8.0)
    back_id = num_foot_nodes * 2 + (2 * i - 1)
    front_id = num_foot_nodes * 2 + (2 * i)
    ops.node(back_id, float(back_x), float(z))
    ops.node(front_id, float(front_x), float(z))
    stem_pairs.append((back_id, front_id))

# Material definition
ops.nDMaterial('ElasticIsotropic', 1, 32000000.0, 0.2)

# Element definition
# Footing: create two triangles per cell between bottom/top nodes
elem_id = 1
for i in range(num_foot_nodes - 1):
    b1 = bottom_node_ids[i]
    b2 = bottom_node_ids[i + 1]
    t1 = top_node_ids[i]
    t2 = top_node_ids[i + 1]
    # triangle 1: b1, t2, t1
    ops.element('Tri31', elem_id, b1, t2, t1, 1.0, 'PlaneStrain', 1)
    elem_id += 1
    # triangle 2: b1, b2, t2
    ops.element('Tri31', elem_id, b1, b2, t2, 1.0, 'PlaneStrain', 1)
    elem_id += 1

# Stem: connect top-of-footing (left_col, right_col) to stem pairs
# find indices of left (x==toe) and front (x==toe+bot_wall) among top_node x positions
top_xs = xs_unique
def find_index_close(val, arr):
    for ii, xx in enumerate(arr):
        if abs(xx - val) < 1e-6:
            return ii
    return None

left_col_idx = find_index_close(toe, top_xs)
front_col_idx = find_index_close(toe + bot_wall, top_xs)
if left_col_idx is None or front_col_idx is None:
    # fallback to using the two central columns if exact matches fail
    left_col_idx = 1
    front_col_idx = 2

left_node = top_node_ids[left_col_idx]
right_node = top_node_ids[front_col_idx]

# build stem elements following the splitting pattern used previously
stem_elements_by_layer = []
for j in range(1, 9):
    b_j, f_j = stem_pairs[j - 1]
    if j == 1:
        # first strip connects left_node, front_j, back_j and right_node, front_j, left_node
        ops.element('Tri31', elem_id, left_node, f_j, b_j, 1.0, 'PlaneStrain', 1)
        e1 = elem_id
        elem_id += 1
        ops.element('Tri31', elem_id, right_node, f_j, left_node, 1.0, 'PlaneStrain', 1)
        e2 = elem_id
        elem_id += 1
    else:
        b_prev, f_prev = stem_pairs[j - 2]
        # element connecting previous back to current back via current front
        ops.element('Tri31', elem_id, b_prev, f_j, b_j, 1.0, 'PlaneStrain', 1)
        e1 = elem_id
        elem_id += 1
        # element connecting previous front, current front, previous back
        ops.element('Tri31', elem_id, f_prev, f_j, b_prev, 1.0, 'PlaneStrain', 1)
        e2 = elem_id
        elem_id += 1
    stem_elements_by_layer.append([e1, e2])

# Shear key mesh (Tri31 3x3 grid) if enabled
if include_shear_key:
    # Shear key protrudes BELOW the footing base (y=0 downward)
    # x range: from shear_key_distance to shear_key_distance + shear_key_width
    base_node_id = num_foot_nodes * 2 + len(stem_pairs) * 2
    shear_node_ids = []
    shear_key_top_nids = []     # top row (y=0) for equalDOF connection
    shear_key_bottom_nids = []  # bottom row (y=-shear_key_thickness) for springs
    shear_key_left_nids = []    # left column for active pressure
    shear_key_right_nids = []   # right column for passive pressure
    nx_sk, ny_sk = 3, 3
    x_start_sk = shear_key_distance
    x_end_sk = shear_key_distance + shear_key_width
    y_top_sk = 0.0              # footing base level
    y_bottom_sk = -shear_key_thickness
    sk_xs = []
    for i_sk in range(nx_sk):
        x = x_start_sk + (x_end_sk - x_start_sk) * i_sk / (nx_sk - 1)
        sk_xs.append(x)
        for j_sk in range(ny_sk):
            y = y_bottom_sk + (y_top_sk - y_bottom_sk) * j_sk / (ny_sk - 1)
            nid = base_node_id + len(shear_node_ids) + 1
            ops.node(nid, float(x), float(y))
            shear_node_ids.append(nid)
            if j_sk == ny_sk - 1:  # top row
                shear_key_top_nids.append(nid)
            if j_sk == 0:  # bottom row
                shear_key_bottom_nids.append(nid)
            if i_sk == 0:  # left column (back face — active side)
                shear_key_left_nids.append(nid)
            if i_sk == nx_sk - 1:  # right column (front face — passive side)
                shear_key_right_nids.append(nid)
    sk_ys_unique = [y_bottom_sk + (y_top_sk - y_bottom_sk) * j_sk / (ny_sk - 1) for j_sk in range(ny_sk)]

    # Create Tri31 elements for the shear key (2 triangles per quad)
    shear_key_elem_ids = []
    for i_sk in range(nx_sk - 1):
        for j_sk in range(ny_sk - 1):
            n1 = shear_node_ids[i_sk * ny_sk + j_sk]
            n2 = shear_node_ids[(i_sk + 1) * ny_sk + j_sk]
            n3 = shear_node_ids[i_sk * ny_sk + (j_sk + 1)]
            n4 = shear_node_ids[(i_sk + 1) * ny_sk + (j_sk + 1)]
            ops.element('Tri31', elem_id, n1, n4, n3, 1.0, 'PlaneStrain', 1)
            shear_key_elem_ids.append(elem_id)
            elem_id += 1
            ops.element('Tri31', elem_id, n1, n2, n4, 1.0, 'PlaneStrain', 1)
            shear_key_elem_ids.append(elem_id)
            elem_id += 1

    # Connect top row of shear key to nearest footing bottom nodes via equalDOF
    for sk_top_nid in shear_key_top_nids:
        sk_x = ops.nodeCoord(sk_top_nid, 1)
        # Find closest bottom footing node
        best_dist = float('inf')
        best_nid = bottom_node_ids[0]
        for bnid in bottom_node_ids:
            bx = ops.nodeCoord(bnid, 1)
            dist = abs(bx - sk_x)
            if dist < best_dist:
                best_dist = dist
                best_nid = bnid
        ops.equalDOF(best_nid, sk_top_nid, 1, 2)

# total nodes and elements
total_nodes = num_foot_nodes * 2 + len(stem_pairs) * 2 + (len(shear_node_ids) if include_shear_key else 0)
total_elements = elem_id - 1

# heel_nodes kept for reference (top-of-footing sampling points)
heel_nodes = top_node_ids

# Defining time series
ops.timeSeries("Linear", 1)

# Solution algorithm
ops.algorithm('Linear')

# Define a load class
# We have only one type of load.
ops.pattern("Plain", 1, 1)

# Active earth pressure coefficient (Rankine)
Ka = np.tan(np.radians(45 - phi/2))**2

# Back face wall nodes (use front nodes of stem pairs as sampling points)
wall_nodes = {}
for i, (b, f) in enumerate(stem_pairs, start=1):
    wall_nodes[f] = Hw * (i / 8.0)

def polygon_centroid(vertices):
    # vertices: list of (x,y) tuples (must be closed or will close)
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return (0.0, 0.0)
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return (Cx, Cy)

def polygon_area_and_centroid(vertices):
    # vertices: list of (x,y) tuples (must be closed or will close)
    if not vertices:
        return 0.0, (None, None)
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return 0.0, (0.0, 0.0)
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return abs(A), (Cx, Cy)

# Build stem polygon (geometry of the wall above top-of-footing)
stem_poly = [(toe, -h_ftg)] + stem_pts[:5] + [(toe + bot_wall, -h_ftg)]
stem_centroid = (None, None)
if stem_poly:
    stem_centroid = polygon_centroid(stem_poly)

# Base (footing slab) polygon (rectangular approximation)
base_poly = [(0.0, 0.0), (float(ftg), 0.0), (float(ftg), -h_ftg), (0.0, -h_ftg)]
base_centroid = polygon_centroid(base_poly)

# Soil (heel block) polygons for drawing and centroids
h_wet_height = min(h_soil, Hwtr)
if taper_direction == 'Toe-facing':
    soil_wet_poly = [(toe + bot_wall, -h_ftg), (float(ftg), -h_ftg), (float(ftg), -(h_ftg + h_wet_height)), (toe + bot_wall, -(h_ftg + h_wet_height))] if h_wet_height > 0 else []
    if h_soil > Hwtr:
        soil_dry_poly = [(toe + bot_wall, -(h_ftg + h_wet_height)), (float(ftg), -(h_ftg + h_wet_height)), (float(ftg), -(h_ftg + h_soil)), (toe + bot_wall, -(h_ftg + h_soil))]
    else:
        soil_dry_poly = []
else:
    # Heel-facing
    x_wet_top = toe + bot_wall - taper * (h_wet_height / Hw) if Hw > 0 else toe + bot_wall
    soil_wet_poly = [(toe + bot_wall, -h_ftg), (float(ftg), -h_ftg), (float(ftg), -(h_ftg + h_wet_height)), (x_wet_top, -(h_ftg + h_wet_height))] if h_wet_height > 0 else []
    if h_soil > Hwtr:
        if h_soil <= Hw:
            x_soil_top = toe + bot_wall - taper * (h_soil / Hw) if Hw > 0 else toe + bot_wall
            soil_dry_poly = [
                (x_wet_top, -(h_ftg + h_wet_height)),
                (float(ftg), -(h_ftg + h_wet_height)),
                (float(ftg), -(h_ftg + h_soil)),
                (x_soil_top, -(h_ftg + h_soil))
            ]
        else:
            x_soil_top = toe + top_wall
            soil_dry_poly = [
                (x_wet_top, -(h_ftg + h_wet_height)),
                (float(ftg), -(h_ftg + h_wet_height)),
                (float(ftg), -(h_ftg + h_soil)),
                (x_soil_top, -(h_ftg + h_soil)),
                (x_soil_top, -(h_ftg + Hw))
            ]
    else:
        soil_dry_poly = []

soil_dry_centroid = polygon_centroid(soil_dry_poly) if soil_dry_poly else (None, None)
soil_wet_centroid = polygon_centroid(soil_wet_poly) if soil_wet_poly else (None, None)

# Map each stem front node to stem centroid x for previous uses
centroid_x_wall = {}
for _, front_id in stem_pairs:
    centroid_x_wall[front_id] = stem_centroid[0] if stem_centroid[0] is not None else (toe + bot_wall) / 2.0

# Compute lateral earth pressure at each node
sigma_total = {}
sigma_soil_q = {}
sigma_w_dict = {}

for node, z in wall_nodes.items():

    # Pressures only exist below h_soil
    if z <= h_soil:
        # Soil pressure (z is elevation from top of footing)
        h_dry_z = max(0.0, h_soil - max(z, Hwtr))
        h_wet_z = max(0.0, min(h_soil, Hwtr) - z)
        sigma_soil = Ka * (gamma_soil_dry * h_dry_z + gamma_soil_wet * h_wet_z)

        # Surcharge pressure
        if surcharge_type == 'Strip Load':
            depth_below_surface = max(1e-5, h_soil - z)
            res_q = stresses_stripload_retainingwall_local(
                imposedstress=q,
                width=width_surcharge,
                offset=offset_surcharge,
                toe_depth=h_soil,
                depth=depth_below_surface
            )
            sigma_q = res_q['delta sigma x [kPa]']
        else:
            sigma_q = Ka * q

        # Water pressure
        if z <= Hwtr:
            sigma_w = gamma_w * (Hwtr - z)
        else:
            sigma_w = 0.0
    else:
        sigma_soil = 0.0
        sigma_q = 0.0
        sigma_w = 0.0

    # Total lateral pressure (kN/m2)
    sigma_total[node] = sigma_soil + sigma_q + sigma_w
    sigma_soil_q[node] = sigma_soil + sigma_q
    sigma_w_dict[node] = sigma_w

# Sort nodes by elevation (z) and compute tributary heights based on centroids
nodes_sorted = sorted(wall_nodes.items(), key=lambda x: x[1])
nodes_order = [n for n, _ in nodes_sorted]
z_list = [z for _, z in nodes_sorted]
Fx = {}
Fx_soil_q = {}
Fx_water_lat = {}
try:
    # build vertical boundaries: bottom = 0, internal midpoints, top = Hw
    n_stem = len(z_list)
    bounds = [0.0]
    for k in range(n_stem - 1):
        bounds.append(0.5 * (z_list[k] + z_list[k + 1]))
    bounds.append(float(Hw))

    for i, node in enumerate(nodes_order):
        h_trib = bounds[i + 1] - bounds[i]
        p_node = sigma_total.get(node, 0.0)
        p_soil_q = sigma_soil_q.get(node, 0.0)
        p_w = sigma_w_dict.get(node, 0.0)
        # lateral force on tributary area (per unit thickness t)
        Fx[node] = -p_node * h_trib * t
        Fx_soil_q[node] = -p_soil_q * h_trib * t
        Fx_water_lat[node] = -p_w * h_trib * t
except Exception:
    # fallback to previous segment approach if anything fails
    Fx = {node: 0.0 for node, _ in nodes_sorted}
    Fx_soil_q = {node: 0.0 for node, _ in nodes_sorted}
    Fx_water_lat = {node: 0.0 for node, _ in nodes_sorted}
    for i in range(len(nodes_sorted)-1):
        n1, z1 = nodes_sorted[i]
        n2, z2 = nodes_sorted[i+1]
        h = z2 - z1
        p1 = sigma_total[n1]
        p2 = sigma_total[n2]
        F = 0.5 * (p1 + p2) * h * t
        Fx[n1] += -0.5 * F
        Fx[n2] += -0.5 * F

        p1_soil_q = sigma_soil_q[n1]
        p2_soil_q = sigma_soil_q[n2]
        F_soil_q = 0.5 * (p1_soil_q + p2_soil_q) * h * t
        Fx_soil_q[n1] += -0.5 * F_soil_q
        Fx_soil_q[n2] += -0.5 * F_soil_q

        p1_w = sigma_w_dict[n1]
        p2_w = sigma_w_dict[n2]
        F_w = 0.5 * (p1_w + p2_w) * h * t
        Fx_water_lat[n1] += -0.5 * F_w
        Fx_water_lat[n2] += -0.5 * F_w

# Apply calculated earth pressure loads
for node, force in Fx.items():
    ops.load(node, force, 0.0)

#VERTICAL LOAD -------------------------------

# Apply concrete self-weight as body force to all elements
for ele in range(1, total_elements + 1):
    ops.eleLoad('-ele', ele, '-type', '-bodyForce', 0.0, -gamma_c)

#soil weight
h_dry = max(0.0, h_soil - Hwtr)
h_wet = Hwtr + h_ftg
q_soil = gamma_soil_dry * h_dry + gamma_soil_wet * h_wet   # kN/m2

# Compute tributary lengths for top-of-footing nodes (heel slab nodes)
# Use centroid-based tributary widths along the footing: domain from x=0 to x=ftg
heel_trib = {}
try:
    xs = xs_unique
    nxs = len(xs)
    # compute shared footing x-boundaries once for both top and bottom nodes
    x_bounds_footing = [0.0]
    for k in range(nxs - 1):
        x_bounds_footing.append(0.5 * (xs[k] + xs[k + 1]))
    x_bounds_footing.append(float(ftg))
    for idx, nid in enumerate(top_node_ids):
        L = x_bounds_footing[idx + 1] - x_bounds_footing[idx]
        heel_trib[nid] = L
except Exception:
    # fallback to neighbor half-distance method
    heel_trib = {}
    for idx, nid in enumerate(top_node_ids):
        if idx == 0:
            L = xs_unique[1] - xs_unique[0]
        elif idx == len(xs_unique) - 1:
            L = xs_unique[-1] - xs_unique[-2]
        else:
            L = 0.5 * (xs_unique[idx] - xs_unique[idx - 1]) + 0.5 * (xs_unique[idx + 1] - xs_unique[idx])
        heel_trib[nid] = L
# Apply soil weight and surcharge to top-of-footing nodes (heel nodes)
for idx, node in enumerate(top_node_ids):
    L = heel_trib[node]
    x_val = xs_unique[idx]
    if x_val >= (toe + bot_wall):
        Fy_soil = -q_soil * L * t
        if surcharge_type == 'Strip Load':
            d_heel = x_val - (toe + bot_wall)
            q_v = stresses_stripload_local(
                z=h_soil,
                x=d_heel - offset_surcharge,
                width=width_surcharge,
                imposedstress=q
            )['delta sigma z [kPa]']
        else:
            q_v = q
        Fy_q = -q_v * L * t
    else:
        Fy_soil = 0.0
        Fy_q = 0.0
    ops.load(node, 0.0, Fy_soil + Fy_q)
bottom_trib = {}
try:
    xs = xs_unique
    nxs = len(xs)
    x_bounds = [0.0]
    for k in range(nxs - 1):
        x_bounds.append(0.5 * (xs[k] + xs[k + 1]))
    x_bounds.append(float(ftg))
    for idx, nid in enumerate(bottom_node_ids):
        L = x_bounds_footing[idx + 1] - x_bounds_footing[idx]
        bottom_trib[nid] = L
except Exception:
    bottom_trib = {}
    for idx, nid in enumerate(bottom_node_ids):
        if idx == 0:
            L = xs_unique[1] - xs_unique[0]
        elif idx == len(xs_unique) - 1:
            L = xs_unique[-1] - xs_unique[-2]
        else:
            L = 0.5 * (xs_unique[idx] - xs_unique[idx - 1]) + 0.5 * (xs_unique[idx + 1] - xs_unique[idx])
        bottom_trib[nid] = L

# compute centroid x positions for footing nodes (centroid of each tributary interval)
centroid_x_base = {}
try:
    for idx, nid in enumerate(top_node_ids):
        cx = 0.5 * (x_bounds_footing[idx] + x_bounds_footing[idx + 1])
        centroid_x_base[nid] = cx
    for idx, nid in enumerate(bottom_node_ids):
        cx = 0.5 * (x_bounds_footing[idx] + x_bounds_footing[idx + 1])
        centroid_x_base[nid] = cx
except Exception:
    centroid_x_base = {}

uplift_pressure = {nid: gamma_w * Hwtr for nid in bottom_node_ids}

# Apply uplift to bottom nodes
for node, L in bottom_trib.items():
    Fy = uplift_pressure.get(node, 0.0) * L * t   # upward force
    ops.load(node, 0.0, Fy)

# Passive soil resistance on toe side footing face (mobilized using FS_passive = 2.0)
FS_passive = 2.0
Kp = np.tan(np.radians(45.0 + phi / 2.0)) ** 2
sigma_p_top = (Kp * gamma_soil_dry * h_soil_toe) / FS_passive
sigma_p_bot = (Kp * gamma_soil_dry * (h_soil_toe + h_ftg)) / FS_passive

F_passive_top = ((2.0 * sigma_p_top + sigma_p_bot) / 6.0) * h_ftg * t
F_passive_bot = ((sigma_p_top + 2.0 * sigma_p_bot) / 6.0) * h_ftg * t

# Apply positive X force on the front face nodes of the footing (toe side)
ops.load(top_node_ids[0], float(F_passive_top), 0.0)
ops.load(bottom_node_ids[0], float(F_passive_bot), 0.0)

# Total passive force (integrated)
P_passive = 0.5 * (sigma_p_top + sigma_p_bot) * h_ftg * t

# Shear key active and passive earth pressure loads
P_passive_shear_key = 0.0
if include_shear_key:
    # The shear key extends from y=0 to y=-shear_key_thickness below footing base
    # Depth below ground surface at footing base = h_soil_toe + h_ftg
    depth_at_ftg_base = h_soil_toe + h_ftg
    # Left face (back side, closer to heel) receives ACTIVE pressure pushing left (negative x)
    # Right face (front side, closer to toe) receives PASSIVE pressure pushing right (positive x)
    sk_ny = len(shear_key_left_nids)
    sk_h_trib = shear_key_thickness / max(1, sk_ny - 1)
    for j_sk, (left_nid, right_nid) in enumerate(zip(shear_key_left_nids, shear_key_right_nids)):
        y_nid = ops.nodeCoord(left_nid, 2)  # y coordinate (negative)
        depth_below_surface = depth_at_ftg_base + abs(y_nid)
        # Tributary height
        if j_sk == 0 or j_sk == sk_ny - 1:
            h_t = sk_h_trib / 2.0
        else:
            h_t = sk_h_trib
        # Active pressure on left face (pushing left = negative x)
        sigma_a = Ka * gamma_soil_dry * depth_below_surface
        F_active = -sigma_a * h_t * t
        ops.load(left_nid, float(F_active), 0.0)
        # Passive pressure on right face (pushing right = positive x) with FS
        sigma_p = (Kp * gamma_soil_dry * depth_below_surface) / FS_passive
        F_passive_sk = sigma_p * h_t * t
        ops.load(right_nid, float(F_passive_sk), 0.0)
        P_passive_shear_key += F_passive_sk
    P_passive += P_passive_shear_key

# Support information - Winkler Soil Spring Model
# Compute subgrade reaction modulus using Vesic's formula: ks = Es / (B * (1 - nu^2))
ks_subgrade = Es_soil / (ftg * (1.0 - nu_soil**2))

# Create spring material for vertical DOF
spring_mat_base_id = 100  # material tag offset for springs
spring_elem_base_id = 1000  # element tag offset for springs

# Create anchor nodes below bottom nodes (fixed) and connect with zero-length springs
anchor_node_base_id = 200  # node tag offset for anchor nodes
for idx, nid in enumerate(bottom_node_ids):
    anchor_nid = anchor_node_base_id + idx
    x_coord = ops.nodeCoord(nid, 1)
    y_coord = ops.nodeCoord(nid, 2)
    ops.node(anchor_nid, float(x_coord), float(y_coord))
    ops.fix(anchor_nid, 1, 1)  # fully fixed anchor node
    
    # Compute spring stiffness = ks * tributary_length * t
    L_trib = bottom_trib.get(nid, ftg / len(bottom_node_ids))
    k_spring = ks_subgrade * L_trib * t
    
    # Create elastic spring material (unique per node to allow different stiffnesses)
    mat_id = spring_mat_base_id + idx
    ops.uniaxialMaterial('Elastic', mat_id, float(k_spring))
    
    # Create zero-length element connecting anchor to bottom node (vertical spring, DOF 2)
    elem_tag = spring_elem_base_id + idx
    ops.element('zeroLength', elem_tag, anchor_nid, nid, '-mat', mat_id, '-dir', 2)

# Fix leftmost bottom node in horizontal DOF only (prevent rigid-body sliding)
leftmost_bottom = bottom_node_ids[0]
ops.fix(leftmost_bottom, 1, 0)

# Shear key bottom node springs (anchor + vertical spring)
if include_shear_key:
    sk_spring_mat_base = 300  # material tag offset for shear key springs
    sk_spring_elem_base = 2000  # element tag offset for shear key springs
    sk_anchor_base = 400  # node tag offset for shear key anchor nodes
    for idx_sk, sk_nid in enumerate(shear_key_bottom_nids):
        anchor_sk_nid = sk_anchor_base + idx_sk
        x_c = ops.nodeCoord(sk_nid, 1)
        y_c = ops.nodeCoord(sk_nid, 2)
        ops.node(anchor_sk_nid, float(x_c), float(y_c))
        ops.fix(anchor_sk_nid, 1, 1)
        # Tributary width for spring
        L_trib_sk = shear_key_width / max(1, len(shear_key_bottom_nids) - 1)
        if idx_sk == 0 or idx_sk == len(shear_key_bottom_nids) - 1:
            L_trib_sk = L_trib_sk / 2.0
        k_spring_sk = ks_subgrade * L_trib_sk * t
        mat_sk = sk_spring_mat_base + idx_sk
        ops.uniaxialMaterial('Elastic', mat_sk, float(k_spring_sk))
        elem_sk = sk_spring_elem_base + idx_sk
        ops.element('zeroLength', elem_sk, anchor_sk_nid, sk_nid, '-mat', mat_sk, '-dir', 2)

ops.system('BandSPD')
ops.numberer('RCM')
ops.constraints('Transformation')
ops.integrator('LoadControl', 1.0)

# Analysis type
ops.analysis('Static')
ops.analyze(1) # Specifies how many times the analysis will be performed.

# ---------------------------------
# Output
# ---------------------------------

# --- Seismic treatment ---
# Option A: Pseudo-static (apply kh*W lateral load)
# --- Seismic treatment (Pseudo-static automatically included) ---
lateral_per_node = 0.0
kh = PGA * FPGA
if kh > 0.0:
    # approximate weights (kN) per unit thickness * thickness `t`
    wall_area = (top_wall + bot_wall) / 2.0 * Hw
    footing_area = ftg * h_ftg
    W_conc = gamma_c * (wall_area + footing_area) * t
    h_dry = max(0.0, h_soil - Hwtr)
    h_wet = Hwtr + h_ftg
    W_soil = (gamma_soil_dry * h_dry + gamma_soil_wet * h_wet) * ftg * t
    W_total = W_soil + W_conc

    total_lateral = kh * W_total
    # distribute equally to back-face wall nodes
    nodes = list(wall_nodes.keys())
    n_w = len(nodes) if len(nodes) > 0 else 1
    lateral_per_node = - total_lateral / n_w

    # Apply as a new plain load pattern so existing loads remain
    pseudo_pattern = 3
    ops.pattern('Plain', pseudo_pattern, 1)
    for node in nodes:
        ops.load(node, lateral_per_node, 0.0)

    # Re-run a static analysis to obtain displacements under combined loads
    ops.system('BandSPD')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    ok_ps = ops.analyze(1)
    if ok_ps == 0:
        max_x = 0.0
        max_y = 0.0
        for i in range(1, total_nodes + 1):
            try:
                dx = abs(ops.nodeDisp(i, 1))
                dy = abs(ops.nodeDisp(i, 2))
                if dx > max_x:
                    max_x = dx
                if dy > max_y:
                    max_y = dy
            except Exception:
                pass
    else:
        st.error('Pseudo-static analysis failed (ops.analyze returned non-zero).')

# Displacements
for i in range(1, total_nodes + 1):
    try:
        print(i, "node x=", ops.nodeDisp(i, 1), "m  -  y=", ops.nodeDisp(i, 2), "m")
    except Exception:
        pass


# =========================================================
# CONSOLIDATED COMPUTATIONS (Stability, Loads, and Metrics)
# =========================================================

# 1. Total lateral (resultant) from Fx dictionary (kN)
try:
    total_lateral_resultant = -sum(Fx.values()) - P_passive  # sign: Fx stored as negative, so -sum(Fx) is positive active. Subtracting P_passive reduces it.
except Exception:
    total_lateral_resultant = 0.0

try:
    soil_node_loads = {}
    for idx, nid in enumerate(top_node_ids):
        L = heel_trib[nid]
        x_val = xs_unique[idx]
        if x_val >= (toe + bot_wall):
            if surcharge_type == 'Strip Load':
                d_heel = x_val - (toe + bot_wall)
                q_v = stresses_stripload_local(
                    z=h_soil,
                    x=d_heel - offset_surcharge,
                    width=width_surcharge,
                    imposedstress=q
                )['delta sigma z [kPa]']
            else:
                q_v = q
            soil_node_loads[nid] = -(q_soil + q_v) * L * t
        else:
            soil_node_loads[nid] = 0.0
    soil_lines = "\n".join([f"- Node {n}: {soil_node_loads[n]:.3f} kN" for n in sorted(soil_node_loads.keys())])
    soil_total = sum(soil_node_loads.values())
except Exception:
    soil_lines = "(not available)"
    soil_total = 0.0
    soil_node_loads = {}

# 3. Uplift applied at bottom nodes (bottom_trib)
try:
    uplift_node_loads = {n: (uplift_pressure.get(n, 0.0) * L * t) for n, L in bottom_trib.items()}
    uplift_lines = "\n".join([f"- Node {n}: {uplift_node_loads[n]:.3f} kN" for n in sorted(uplift_node_loads.keys())])
    uplift_total = sum(uplift_node_loads.values())
except Exception:
    uplift_lines = "(not available)"
    uplift_total = 0.0
    uplift_node_loads = {}

net_vertical = soil_total + uplift_total

# 4. Concrete weights and lever arms
base_area = ftg * h_ftg
W_base = gamma_c * base_area * t
cg_x_base = ftg / 2.0

stem_rect_area = top_wall * Hw
W_stem_rect = gamma_c * stem_rect_area * t
if taper_direction == 'Toe-facing':
    cg_x_stem_rect = toe + bot_wall - top_wall / 2.0
else:
    cg_x_stem_rect = toe + top_wall / 2.0

stem_tri_area = 0.5 * taper * Hw
W_stem_tri = gamma_c * stem_tri_area * t
if taper_direction == 'Toe-facing':
    cg_x_stem_tri = toe + (2.0 / 3.0) * taper if taper > 0 else float(toe)
else:
    cg_x_stem_tri = toe + top_wall + taper / 3.0 if taper > 0 else float(toe)

W_conc = W_base + W_stem_rect + W_stem_tri
# Add shear key weight if enabled
W_shear_key = 0.0
if include_shear_key:
    shear_key_area = shear_key_width * shear_key_thickness
    W_shear_key = gamma_c * shear_key_area * t
    W_conc += W_shear_key
wall_area = stem_rect_area + stem_tri_area
footing_area = base_area

# 5. Soil weight on heel and lever arm
A_wet, (cg_x_wet, cg_y_wet) = polygon_area_and_centroid(soil_wet_poly)
A_dry, (cg_x_dry, cg_y_dry) = polygon_area_and_centroid(soil_dry_poly)

W_soil_dry = gamma_soil_dry * A_dry * t
W_soil_wet = gamma_soil_wet * A_wet * t
W_soil = W_soil_dry + W_soil_wet
if W_soil > 0:
    cg_x_soil = (W_soil_dry * cg_x_dry + W_soil_wet * cg_x_wet) / (W_soil_dry + W_soil_wet)
else:
    cg_x_soil = toe + bot_wall + (ftg - (toe + bot_wall)) / 2.0
soil_width = ftg - (toe + bot_wall)

# 5.5 Surcharge vertical weight and lever arm on heel footing
if surcharge_type == 'Strip Load':
    n_pts = 10
    forces_pts = []
    for k in range(n_pts):
        frac = (k + 0.5) / n_pts
        d_pt = frac * soil_width
        x_rel = d_pt - offset_surcharge
        res = stresses_stripload_local(z=h_soil, x=x_rel, width=width_surcharge, imposedstress=q)
        forces_pts.append(max(0.0, res['delta sigma z [kPa]']))
    avg_q_v = np.mean(forces_pts)
else:
    avg_q_v = q
W_surcharge = avg_q_v * soil_width * t

# 6. Global stability checks (about x_pivot = 0.0)
x_pivot_global = 0.0
try:
    M_ot_global = 0.0
    for n, f in Fx.items():
        try:
            coord = ops.nodeCoord(n)
            z_n = float(coord[1])
        except Exception:
            z_n = 0.0
        M_ot_global += abs(f) * abs(z_n)
except Exception:
    M_ot_global = 0.0

try:
    M_uplift_global = uplift_total * (ftg / 2.0 - x_pivot_global)
except Exception:
    M_uplift_global = 0.0

M_res_global = (W_base * abs(cg_x_base - x_pivot_global) +
                W_stem_rect * abs(cg_x_stem_rect - x_pivot_global) +
                W_stem_tri * abs(cg_x_stem_tri - x_pivot_global) +
                (W_soil + W_surcharge) * abs(cg_x_soil - x_pivot_global) -
                M_uplift_global)

try:
    W_down_global = W_conc + W_soil + W_surcharge - uplift_total
except Exception:
    W_down_global = W_conc + W_soil + W_surcharge

try:
    FS_ot_global = M_res_global / M_ot_global if M_ot_global > 0 else float('inf')
except Exception:
    FS_ot_global = None

try:
    phi_rad = np.radians(phi)
    # R_slide is the base friction resistance (W * tan(phi)) + shear key passive resistance
    R_slide = max(0.0, W_down_global) * np.tan(phi_rad) + P_passive_shear_key
    # F_drive is the net driving lateral force (Active + Surcharge + Hydrostatic - Passive)
    # Since passive resistance is applied in OpenSees as a positive horizontal load,
    # total_lateral_resultant (-sum(Fx)) naturally subtracts it.
    F_drive = max(0.0, total_lateral_resultant)
    FS_slide_global = (R_slide + P_passive) / F_drive if F_drive > 0 else float('inf')
except Exception:
    R_slide = 0.0
    FS_slide_global = None

# 7. Stem stability checks (about x_pivot = toe)
x_pivot_stem = float(toe)
x_centroid = float(ftg) / 2.0
try:
    M_ot_stem = 0.0
    for n, f in Fx.items():
        if n in centroid_x_wall:
            x_n = centroid_x_wall[n]
        elif n in centroid_x_base:
            x_n = centroid_x_base[n]
        else:
            try:
                coord = ops.nodeCoord(n)
                x_n = float(coord[0])
            except Exception:
                x_n = x_centroid
        M_ot_stem += (abs(f)) * abs(x_n - x_pivot_stem)
except Exception:
    M_ot_stem = 0.0

try:
    W_down_stem = W_conc + W_soil - uplift_total
except Exception:
    W_down_stem = W_conc + W_soil
M_res_stem = W_down_stem * abs(x_centroid - x_pivot_stem)
try:
    FS_ot_stem = M_res_stem / M_ot_stem if M_ot_stem > 0 else float('inf')
except Exception:
    FS_ot_stem = None

# 8. Seismic coefficients and peak estimates
kh = PGA * FPGA
F_seismic_pseudo = kh * (W_soil + W_conc)
g_acc = 9.81
peak_accel = PGA * g_acc * FPGA
inertial_peak = (W_soil + W_conc) * (peak_accel / g_acc)

# 9. Get maximum displacements
max_disp_x = 0.0
max_disp_y = 0.0
for i in range(1, total_nodes + 1):
    try:
        dx = abs(ops.nodeDisp(i, 1))
        dy = abs(ops.nodeDisp(i, 2))
        if dx > max_disp_x:
            max_disp_x = dx
        if dy > max_disp_y:
            max_disp_y = dy
    except Exception:
        pass

# 9b. FEM-Based Settlement Analysis (from soil spring model)
try:
    # Toe settlement (leftmost bottom node) - y displacement (downward is negative)
    disp_toe_x = ops.nodeDisp(bottom_node_ids[0], 1)
    disp_toe_y = ops.nodeDisp(bottom_node_ids[0], 2)
    settlement_toe_mm = abs(disp_toe_y) * 1000.0  # convert m to mm
    
    # Heel settlement (rightmost bottom node)
    disp_heel_x = ops.nodeDisp(bottom_node_ids[-1], 1)
    disp_heel_y = ops.nodeDisp(bottom_node_ids[-1], 2)
    settlement_heel_mm = abs(disp_heel_y) * 1000.0
    
    # Differential settlement
    diff_settlement_mm = abs(disp_toe_y - disp_heel_y) * 1000.0
    
    # Rotation (radians and degrees)
    dist_toe_heel = abs(xs_unique[-1] - xs_unique[0])
    if dist_toe_heel > 1e-6:
        rotation_rad = np.arctan(abs(disp_toe_y - disp_heel_y) / dist_toe_heel)
    else:
        rotation_rad = 0.0
    rotation_deg = np.degrees(rotation_rad)
    
    # Allowable differential settlement per SNI 8460:2017 = 50 mm
    allowable_diff_settlement_mm = 50.0
    diff_settlement_pass = diff_settlement_mm <= allowable_diff_settlement_mm
    
    # Build bottom node displacement table
    bottom_disp_rows = []
    for nid in bottom_node_ids:
        try:
            dx = ops.nodeDisp(nid, 1)
            dy = ops.nodeDisp(nid, 2)
            x_pos = ops.nodeCoord(nid, 1)
            bottom_disp_rows.append({
                'Node': nid,
                'X Position (m)': round(x_pos, 3),
                'Disp X (mm)': round(dx * 1000, 4),
                'Disp Y (mm)': round(dy * 1000, 4),
                'Settlement (mm)': round(abs(dy) * 1000, 4)
            })
        except Exception:
            pass
    df_bottom_disp = pd.DataFrame(bottom_disp_rows)
except Exception:
    settlement_toe_mm = 0.0
    settlement_heel_mm = 0.0
    diff_settlement_mm = 0.0
    rotation_rad = 0.0
    rotation_deg = 0.0
    diff_settlement_pass = True
    df_bottom_disp = pd.DataFrame()

# 10. Generate Summary DataFrames
df_lateral = pd.DataFrame()
df_vertical = pd.DataFrame()
try:
    # Lateral table (wall nodes)
    lateral_rows = []
    total_lat_force = 0.0
    total_lat_moment = 0.0
    for n in sorted(Fx.keys()):
        f = Fx[n]
        try:
            coord = ops.nodeCoord(n)
            z_n = float(coord[1])
        except Exception:
            z_n = 0.0
        moment = abs(f) * abs(z_n)
        lateral_rows.append({'Node': n, 'Y (m) lever': z_n, 'Force (kN)': f, 'Moment (kN·m)': moment})
        total_lat_force += f
        total_lat_moment += moment

    df_lateral = pd.DataFrame(lateral_rows)
    df_lateral = pd.concat([df_lateral, pd.DataFrame([{'Node': 'TOTAL', 'Y (m) lever': np.nan, 'Force (kN)': total_lat_force, 'Moment (kN·m)': total_lat_moment}])], ignore_index=True)
    df_lateral['Node'] = df_lateral['Node'].astype(str)

    # Vertical table (top nodes soil and bottom uplift)
    vertical_rows = []
    total_vert_force = 0.0
    total_vert_moment = 0.0
    vert_nodes = set(list(heel_trib.keys()) + list(bottom_trib.keys()))
    for n in sorted(vert_nodes):
        soil_f = abs(soil_node_loads.get(n, 0.0))
        uplift_f = -abs(uplift_node_loads.get(n, 0.0))
        f_vert = soil_f + uplift_f
        if n in centroid_x_base:
            x_n = centroid_x_base[n]
        else:
            try:
                coord = ops.nodeCoord(n)
                x_n = float(coord[0])
            except Exception:
                x_n = float(ftg) / 2.0
        moment = f_vert * abs(x_n - x_pivot_global)
        vertical_rows.append({'Node': n, 'X (m)': x_n, 'Soil (kN)': soil_node_loads.get(n, 0.0), 'Uplift (kN)': uplift_node_loads.get(n, 0.0), 'Total Vertical (kN)': f_vert, 'Moment (kN·m)': moment})
        total_vert_force += f_vert
        total_vert_moment += moment

    df_vertical = pd.DataFrame(vertical_rows)
    df_vertical = pd.concat([df_vertical, pd.DataFrame([{'Node': 'TOTAL', 'X (m)': np.nan, 'Soil (kN)': np.nan, 'Uplift (kN)': np.nan, 'Total Vertical (kN)': total_vert_force, 'Moment (kN·m)': total_vert_moment}])], ignore_index=True)
    df_vertical['Node'] = df_vertical['Node'].astype(str)
except Exception as e:
    pass

# Helper to get stem moments
def get_stem_moments():
    node_moments = {}
    
    # 1. Base of the stem
    base_nodes = [left_node, right_node]
    M_base = 0.0
    centroid_base = (ops.nodeCoord(left_node, 1) + ops.nodeCoord(right_node, 1))/2.0
    for ele in stem_elements_by_layer[0]:
        nodes = ops.eleNodes(ele)
        forces = ops.eleResponse(ele, 'forces')
        for i, node in enumerate(nodes):
            if node in base_nodes:
                Fy = forces[i*2 + 1]
                x = ops.nodeCoord(node, 1)
                M_base -= Fy * (x - centroid_base)
    for node in base_nodes:
        node_moments[node] = M_base
    
    # 2. Layer cuts (1 to 8)
    for j in range(1, 9):
        b_j, f_j = stem_pairs[j-1]
        cut_nodes = [b_j, f_j]
        M_cut = 0.0
        centroid_cut = (ops.nodeCoord(b_j, 1) + ops.nodeCoord(f_j, 1))/2.0
        for ele in stem_elements_by_layer[j-1]:
            nodes = ops.eleNodes(ele)
            forces = ops.eleResponse(ele, 'forces')
            for i, node in enumerate(nodes):
                if node in cut_nodes:
                    Fy = forces[i*2 + 1]
                    x = ops.nodeCoord(node, 1)
                    M_cut += Fy * (x - centroid_cut)
        for node in cut_nodes:
            node_moments[node] = M_cut
        
    return node_moments


# =========================================================
# CONCRETE PROPERTIES ULTIMATE BENDING CAPACITY CODE
# =========================================================

# Concrete Stress Block parameters (AS3600 approximation)
alpha_stress = max(0.67, min(0.85, 0.85 - 0.0015 * fc))
gamma_stress = max(0.67, min(0.85, 0.97 - 0.0025 * fc))

try:
    from concreteproperties.material import Concrete, SteelBar
    import concreteproperties.stress_strain_profile as ssp
    from concreteproperties.pre import add_bar_rectangular_array
    from concreteproperties.concrete_section import ConcreteSection
    from sectionproperties.pre.library.primitive_sections import rectangular_section
    
    # Define Materials
    material_concrete = Concrete(
        name="Concrete Class",
        density=2.4e-6,
        stress_strain_profile=ssp.ConcreteLinear(elastic_modulus=Ec),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
            compressive_strength=fc,
            alpha=alpha_stress,
            gamma=gamma_stress,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=ft_tensile,
        colour="lightgrey",
    )
    
    material_steel = SteelBar(
        name="Steel Class",
        density=7.85e-6,
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=fy,
            elastic_modulus=Es,
            fracture_strain=0.05,
        ),
        colour="grey",
    )
    
    # 1. Stem Section Bending Capacity (1m wide strip)
    stem_beam_width = 1000.0  # mm
    stem_beam_height = bot_wall * 1000.0  # mm
    
    geom_stem = rectangular_section(d=stem_beam_height, b=stem_beam_width, material=material_concrete)
    n_x_bars_stem = int((stem_beam_width - 2 * stem_cover) / stem_spacing_x) + 1
    
    # Bottom bars
    geom_stem = add_bar_rectangular_array(
        geometry=geom_stem,
        area=stem_bar_area,
        material=material_steel,
        n_x=n_x_bars_stem,
        x_s=stem_spacing_x,
        anchor=(stem_cover, stem_cover),
    )
    
    # Top bars
    geom_stem = add_bar_rectangular_array(
        geometry=geom_stem,
        area=stem_bar_area,
        material=material_steel,
        n_x=n_x_bars_stem,
        x_s=stem_spacing_x,
        anchor=(stem_cover, stem_beam_height - stem_cover),
    )
    
    # Side bars (evenly spaced between top and bottom main bars, avoiding out-of-bounds/overlap)
    n_y_stem = int((stem_beam_height - 2 * stem_cover - 1e-3) / stem_spacing_y)
    if n_y_stem > 0:
        geom_stem = add_bar_rectangular_array(
            geometry=geom_stem,
            area=stem_sidebar_area,
            material=material_steel,
            n_x=2,
            x_s=stem_beam_width - 2 * stem_cover,
            n_y=n_y_stem,
            y_s=stem_spacing_y,
            anchor=(stem_cover, stem_cover + stem_spacing_y),
        )
    
    conc_sec_stem = ConcreteSection(geom_stem)
    stem_capacity_sag = conc_sec_stem.ultimate_bending_capacity(theta=0)
    stem_capacity_hog = conc_sec_stem.ultimate_bending_capacity(theta=np.pi)
    
    M_u_stem_allow = min(stem_capacity_sag.m_xy, stem_capacity_hog.m_xy) / 1e6  # kN.m
    
    # Generate Stem section geometry plot
    plt.close('all')
    conc_sec_stem.plot_section()
    fig_stem_sec = plt.gcf()
    fig_stem_sec.set_size_inches(6, 4)
    # Set labels
    try:
        ax = fig_stem_sec.axes[0]
        ax.set_title("Stem Reinforcement Cross Section (1m Width)", fontsize=10)
    except Exception:
        pass
    fig_stem_sec.tight_layout()
    plt.close(fig_stem_sec)

    # 2. Footing Section Bending Capacity (1m wide strip)
    ftg_beam_width = 1000.0  # mm
    ftg_beam_height = h_ftg * 1000.0  # mm
    
    geom_ftg = rectangular_section(d=ftg_beam_height, b=ftg_beam_width, material=material_concrete)
    n_x_bars_ftg = int((ftg_beam_width - 2 * ftg_cover) / ftg_spacing_x) + 1
    n_y_bars_ftg = int((ftg_beam_height - 2 * ftg_cover) / ftg_spacing_y) + 1
    
    # Bottom bars
    geom_ftg = add_bar_rectangular_array(
        geometry=geom_ftg,
        area=ftg_bar_area,
        material=material_steel,
        n_x=n_x_bars_ftg,
        x_s=ftg_spacing_x,
        anchor=(ftg_cover, ftg_cover),
    )
    
    # Top bars
    geom_ftg = add_bar_rectangular_array(
        geometry=geom_ftg,
        area=ftg_bar_area,
        material=material_steel,
        n_x=n_x_bars_ftg,
        x_s=ftg_spacing_x,
        anchor=(ftg_cover, ftg_beam_height - ftg_cover),
    )
    
    # Side bars (evenly spaced between top and bottom main bars, avoiding out-of-bounds/overlap)
    n_y_ftg = int((ftg_beam_height - 2 * ftg_cover - 1e-3) / ftg_spacing_y)
    if n_y_ftg > 0:
        geom_ftg = add_bar_rectangular_array(
            geometry=geom_ftg,
            area=ftg_sidebar_area,
            material=material_steel,
            n_x=2,
            x_s=ftg_beam_width - 2 * ftg_cover,
            n_y=n_y_ftg,
            y_s=ftg_spacing_y,
            anchor=(ftg_cover, ftg_cover + ftg_spacing_y),
        )
    
    conc_sec_ftg = ConcreteSection(geom_ftg)
    ftg_capacity_sag = conc_sec_ftg.ultimate_bending_capacity(theta=0)
    ftg_capacity_hog = conc_sec_ftg.ultimate_bending_capacity(theta=np.pi)
    
    M_u_ftg_sag_allow = ftg_capacity_sag.m_xy / 1e6  # kN.m (sagging, tension at bottom)
    M_u_ftg_hog_allow = ftg_capacity_hog.m_xy / 1e6  # kN.m (hogging, tension at top)
    
    # Generate Footing section geometry plot
    plt.close('all')
    conc_sec_ftg.plot_section()
    fig_ftg_sec = plt.gcf()
    fig_ftg_sec.set_size_inches(6, 4)
    try:
        ax = fig_ftg_sec.axes[0]
        ax.set_title("Footing Reinforcement Cross Section (1m Width)", fontsize=10)
    except Exception:
        pass
    fig_ftg_sec.tight_layout()
    plt.close(fig_ftg_sec)

except Exception as e:
    st.sidebar.error(f"Error calculating bending capacity: {e}")
    M_u_stem_allow = 0.0
    M_u_ftg_sag_allow = 0.0
    M_u_ftg_hog_allow = 0.0
    fig_stem_sec = None
    fig_ftg_sec = None

# ----------------------------------------------------
# Actual Moment Demands Calculations
# ----------------------------------------------------
try:
    # Stem Wall actual moment from FEM base moment
    node_moments_dict = get_stem_moments()
    M_actual_stem = abs(node_moments_dict[left_node]) / t
except Exception:
    M_actual_stem = 0.0

try:
    B_len = ftg
    Rv_force = max(0.1, W_conc + W_soil + W_surcharge - uplift_total)
    M_net_moment = M_res_global - M_ot_global
    x_R_loc = M_net_moment / Rv_force
    e_ecc = B_len / 2.0 - x_R_loc

    if abs(e_ecc) <= B_len / 6.0:
        q_t = (Rv_force / B_len) * (1.0 + 6.0 * e_ecc / B_len)
        q_h = (Rv_force / B_len) * (1.0 - 6.0 * e_ecc / B_len)
    elif e_ecc > B_len / 6.0:
        q_t = (2.0 * Rv_force) / (3.0 * x_R_loc)
        q_h = 0.0
    else:
        q_h = (2.0 * Rv_force) / (3.0 * (B_len - x_R_loc))
        q_t = 0.0

    def get_q_bearing_at(x):
        if abs(e_ecc) <= B_len / 6.0:
            return q_t + (q_h - q_t) * (x / B_len)
        elif e_ecc > B_len / 6.0:
            L_b = 3.0 * x_R_loc
            return q_t * (1.0 - x / L_b) if x <= L_b else 0.0
        else:
            L_b = 3.0 * (B_len - x_R_loc)
            return q_h * (x - (B_len - L_b)) / L_b if x >= B_len - L_b else 0.0

    q_toe_end = get_q_bearing_at(toe)
    q_heel_start = get_q_bearing_at(toe + bot_wall)

    # Toe actual moment (tension at bottom, sagging)
    w_down_t = (gamma_c * h_ftg + gamma_soil_dry * h_soil_toe) * t
    M_bearing_t = (get_q_bearing_at(0.0) * (toe**2) / 3.0 + q_toe_end * (toe**2) / 6.0) * t
    M_down_t = w_down_t * (toe**2) / 2.0
    M_actual_toe = max(0.0, M_bearing_t - M_down_t) / t

    # Heel actual moment (tension at top, hogging)
    w_down_h = (gamma_c * h_ftg + (gamma_soil_dry * h_dry + gamma_soil_wet * h_wet) / B_len * soil_width + q) * t
    heel_len = B_len - (toe + bot_wall)
    M_bearing_h = (q_heel_start * (heel_len**2) / 6.0 + get_q_bearing_at(B_len) * (heel_len**2) / 3.0) * t
    M_down_h = w_down_h * (heel_len**2) / 2.0
    M_actual_heel = max(0.0, M_down_h - M_bearing_h) / t
    sigma_max = max(q_t, q_h)
except Exception:
    M_actual_toe = 0.0
    M_actual_heel = 0.0
    sigma_max = 0.0

# Compute overburden stresses and shear force at footing base (used by bearing and pile analysis)
D_depth = h_ftg + h_soil_toe
H_water_front = h_ftg + Hwtr_front if Hwtr_front > 0 else 0.0
if H_water_front >= D_depth:
    p0_eff = D_depth * (gamma_soil_wet - gamma_w)
    base_sigma_v = D_depth * gamma_soil_wet
elif H_water_front > 0:
    p0_eff = H_water_front * (gamma_soil_wet - gamma_w) + (D_depth - H_water_front) * gamma_soil_dry
    base_sigma_v = H_water_front * gamma_soil_wet + (D_depth - H_water_front) * gamma_soil_dry
else:
    p0_eff = D_depth * gamma_soil_dry
    base_sigma_v = D_depth * gamma_soil_dry

p0_eff = max(0.0, p0_eff)
base_sigma_v = max(0.0, base_sigma_v)
H_force = abs(F_drive) if 'F_drive' in locals() else 0.0

# ----------------------------------------------------
# Bearing Capacity Analysis using Groundhog
# ----------------------------------------------------
q_ult = 0.0
q_allow = 0.0
FS_actual_bearing = float('inf')
bearing_pass = True

if enable_bearing:
    try:
        import groundhog.shallowfoundations.capacity as cap
        
        # Calculate B' and parameters for groundhog
        B_prime = max(0.01, ftg - 2.0 * abs(e_ecc)) if 'e_ecc' in locals() else ftg
        L_prime = 1e5

        # Unit weight below foundation base
        if Hwtr_front > 0 or Hwtr > 0:
            effective_unit_weight_below = max(3.0, gamma_soil_wet - gamma_w)
        else:
            effective_unit_weight_below = gamma_soil_dry

        # Horizontal and vertical forces
        V_force = max(0.1, W_down_global) if 'W_down_global' in locals() else 0.1

        if bearing_soil_type == 'Sand (Drained)':
            load_inc = np.degrees(np.arctan(H_force / V_force))
            # Call groundhog drained capacity
            res = cap.verticalcapacity_drained_api(
                vertical_effective_stress=p0_eff,
                effective_friction_angle=float(phi),
                effective_unit_weight=effective_unit_weight_below,
                effective_length=L_prime,
                effective_width=B_prime,
                base_depth=D_depth,
                skirted=False,
                load_inclination=float(load_inc),
                validate=False
            )
            q_ult = res.get('qu [kPa]', 0.0)
            if np.isnan(q_ult):
                q_ult = 0.0
        else:
            # Clay (Undrained)
            res = cap.verticalcapacity_undrained_api(
                effective_length=L_prime,
                effective_width=B_prime,
                su_base=su_val,
                base_depth=D_depth,
                skirted=False,
                base_sigma_v=base_sigma_v,
                horizontal_load=H_force * L_prime,
                validate=False
            )
            q_ult = res.get('qu [kPa]', 0.0)
            if np.isnan(q_ult):
                q_ult = 0.0

        q_allow = q_ult / FS_bearing
        if sigma_max > 0:
            FS_actual_bearing = q_ult / sigma_max
        else:
            FS_actual_bearing = float('inf')
            
        bearing_pass = FS_actual_bearing >= FS_bearing
    except Exception as e:
        q_ult = 0.0
        q_allow = 0.0
        FS_actual_bearing = 0.0
        bearing_pass = False



# ----------------------------------------------------
# Consolidation Settlement Analysis using Groundhog
# ----------------------------------------------------
consol_settlement_m = 0.0
consol_settlement_mm = 0.0
consol_allow_mm = 0.0
consol_pass = True

if enable_bearing:
    try:
        def primaryconsolidationsettlement_nc(initial_height, initial_voidratio, initial_effective_stress, effective_stress_increase, compression_index, e_min=0.3, **kw):
            de = min(initial_voidratio - e_min, compression_index * np.log10((initial_effective_stress + effective_stress_increase) / initial_effective_stress))
            return {'delta z [m]': (initial_height / (1.0 + initial_voidratio)) * de}

        def primaryconsolidationsettlement_oc(initial_height, initial_voidratio, initial_effective_stress, preconsolidation_pressure, effective_stress_increase, compression_index, recompression_index, e_min=0.3, **kw):
            s_fin = initial_effective_stress + effective_stress_increase
            pc = max(initial_effective_stress, preconsolidation_pressure)
            de = recompression_index * np.log10(s_fin / initial_effective_stress) if s_fin < pc else recompression_index * np.log10(pc / initial_effective_stress) + compression_index * np.log10(s_fin / pc)
            return {'delta z [m]': (initial_height / (1.0 + initial_voidratio)) * min(initial_voidratio - e_min, de)}
        
        # Effective stress increase at foundation base = net bearing pressure
        delta_sigma_consol = max(0.0, sigma_max)
        
        if consol_type == 'Normally Consolidated (NC)':
            res_consol = primaryconsolidationsettlement_nc(
                initial_height=H0_consol,
                initial_voidratio=e0_consol,
                initial_effective_stress=max(1.0, p0_eff),
                effective_stress_increase=delta_sigma_consol,
                compression_index=Cc_consol,
                validate=False
            )
        else:
            res_consol = primaryconsolidationsettlement_oc(
                initial_height=H0_consol,
                initial_voidratio=e0_consol,
                initial_effective_stress=max(1.0, p0_eff),
                preconsolidation_pressure=pc_consol,
                effective_stress_increase=delta_sigma_consol,
                compression_index=Cc_consol,
                recompression_index=Cr_consol,
                validate=False
            )
        
        consol_settlement_m = abs(res_consol.get('delta z [m]', 0.0))
        consol_settlement_mm = consol_settlement_m * 1000.0
        
        # Allowable settlement per SNI 8460:2017: 15 cm + B(cm)/600
        B_cm = ftg * 100.0  # convert m to cm
        consol_allow_cm = 15.0 + B_cm / 600.0
        consol_allow_mm = consol_allow_cm * 10.0  # convert cm to mm
        
        consol_pass = consol_settlement_mm <= consol_allow_mm
    except Exception as e:
        consol_settlement_m = 0.0
        consol_settlement_mm = 0.0
        consol_allow_mm = 0.0
        consol_pass = True

# ----------------------------------------------------
# Base Anchor (Pile) Foundation Analysis
# ----------------------------------------------------
pile_pass_toe_axial = True
pile_pass_toe_lateral = True
pile_pass_heel_axial = True
pile_pass_heel_lateral = True
pile_pass_toe_interaction = True
pile_pass_heel_interaction = True
pile_pass_toe_settlement = True
pile_pass_heel_settlement = True
settlement_toe_pile = 0.0
settlement_heel_pile = 0.0
allowable_pile_settlement_mm = 0.0
fig_pile_sec = None

update_splash(splash_placeholder, 35, "Running 1D OpenSees Stem Beam Analysis", "Solving stem node displacements, bending moments, & shear forces...")

def run_1d_stem_beam_analysis(Hw, top_wall, bot_wall, h_ftg, Ec, h_soil, h_soil_toe,
                               gamma_soil_dry, gamma_soil_wet, phi, q, surcharge_type,
                               width_surcharge, offset_surcharge, Hwtr, Hwtr_front,
                               gamma_w, kh, gamma_c):
    import subprocess
    import sys
    import json

    subprocess_code = f"""
import openseespy.opensees as ops
import numpy as np
import json

def stresses_stripload_retainingwall_local(imposedstress, width, offset, toe_depth, depth):
    H0 = max(1e-5, float(toe_depth))
    z = max(1e-5, float(depth))
    a = float(offset)
    B = float(width)
    
    alpha_rad = np.arctan(a / z)
    alpha_far_rad = np.arctan((a + B) / z)
    beta_rad = alpha_far_rad - alpha_rad
    alpha_bisector_rad = alpha_rad + beta_rad / 2.0
    
    _delta_sigma_x = (2.0 * imposedstress / np.pi) * (
        beta_rad - np.sin(beta_rad) * np.cos(2.0 * alpha_bisector_rad)
    )
    return {{
        'delta sigma x [kPa]': _delta_sigma_x
    }}

# Input arguments passed from main script
Hw = {Hw}
top_wall = {top_wall}
bot_wall = {bot_wall}
h_ftg = {h_ftg}
Ec = {Ec}
h_soil = {h_soil}
h_soil_toe = {h_soil_toe}
gamma_soil_dry = {gamma_soil_dry}
gamma_soil_wet = {gamma_soil_wet}
phi = {phi}
q = {q}
surcharge_type = {repr(surcharge_type)}
width_surcharge = {width_surcharge}
offset_surcharge = {offset_surcharge}
Hwtr = {Hwtr}
Hwtr_front = {Hwtr_front}
gamma_w = {gamma_w}
kh = {kh}
gamma_c = {gamma_c}

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# 9 nodes from y = 0.0 to y = Hw
n_steps = 8
ys = np.linspace(0.0, Hw, n_steps + 1)
for i, y in enumerate(ys, 1):
    ops.node(i, 0.0, float(y))

# Boundary condition: base fixed
ops.fix(1, 1, 1, 1)

# Materials and sections
h_avg = (top_wall + bot_wall) / 2.0
A = h_avg
I = (h_avg ** 3) / 12.0

ops.geomTransf('Linear', 1)
for i in range(1, n_steps + 1):
    ops.element('elasticBeamColumn', i, i, i + 1, A, Ec, I, 1)

# Pressures and forces
Ka = np.tan(np.radians(45.0 - phi / 2.0)) ** 2
Kp = np.tan(np.radians(45.0 + phi / 2.0)) ** 2

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

# Front soil height relative to stem base
h_front_soil_stem = max(0.0, h_soil_toe - h_ftg)
# Front water height relative to stem base
h_wtr_front_stem = max(0.0, Hwtr_front - h_ftg)

for i, z in enumerate(ys, 1):
    # Tributary height
    if i == 1 or i == n_steps + 1:
        h_trib = Hw / (2.0 * n_steps)
    else:
        h_trib = Hw / n_steps
        
    # 1. Active Pressure (Rear)
    if z <= h_soil:
        h_dry_z = max(0.0, h_soil - max(z, Hwtr))
        h_wet_z = max(0.0, min(h_soil, Hwtr) - z)
        sigma_soil = Ka * (gamma_soil_dry * h_dry_z + gamma_soil_wet * h_wet_z)
        
        if surcharge_type == 'Strip Load':
            depth_below_surface = max(1e-5, h_soil - z)
            sigma_q = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, depth_below_surface)['delta sigma x [kPa]']
        else:
            sigma_q = Ka * q
            
        if z <= Hwtr:
            sigma_w = gamma_w * (Hwtr - z)
        else:
            sigma_w = 0.0
            
        p_active = sigma_soil + sigma_q + sigma_w
    else:
        p_active = 0.0
        
    # 2. Passive Pressure (Front)
    if z <= h_front_soil_stem:
        d_front = h_front_soil_stem - z
        sigma_passive = Kp * gamma_soil_dry * d_front # assuming dry soil above toe
        
        if z <= h_wtr_front_stem:
            sigma_w_front = gamma_w * (h_wtr_front_stem - z)
        else:
            sigma_w_front = 0.0
            
        p_passive = sigma_passive + sigma_w_front
    else:
        p_passive = 0.0
        
    # Net lateral pressure (pushing right, positive x)
    p_net = p_passive - p_active
    F_lat = p_net * h_trib
    
    # 3. Concrete Self-Weight and Seismic
    thickness_z = bot_wall - (bot_wall - top_wall) * (z / Hw)
    W_c = gamma_c * thickness_z * h_trib
    F_vert = -W_c
    F_seismic = -kh * W_c  # pushes left (negative x)
    
    # Apply loads
    ops.load(i, float(F_lat + F_seismic), float(F_vert), 0.0)

# Run static analysis
ops.system('BandSPD')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 1.0)
ops.analysis('Static')
ok = ops.analyze(1)

# Extract results
disps = []
shears = []
moments = []
for i in range(1, n_steps + 2):
    disps.append(ops.nodeDisp(i, 1))

for i in range(1, n_steps + 1):
    forces = ops.eleForce(i)
    shears.append(forces[1])
    moments.append(forces[2])
# Append last element forces
forces_last = ops.eleForce(n_steps)
shears.append(-forces_last[4])
moments.append(-forces_last[5])

results = {{
    "ok": int(ok),
    "ys": list(ys),
    "disps": disps,
    "shears": shears,
    "moments": moments
}}
print(json.dumps(results))
"""

    try:
        res = subprocess.run([sys.executable, '-c', subprocess_code], capture_output=True, text=True)
        if res.returncode == 0:
            data = json.loads(res.stdout.strip().split('\\n')[-1])
            return data
        else:
            print("Subprocess failed inside run_1d_stem_beam_analysis:", res.stderr)
            return None
    except Exception as e:
        print("Error during run_1d_stem_beam_analysis execution:", e)
        return None

def run_2d_ssrm_analysis(toe, heel, bot_wall, top_wall, h_ftg, Hw, h_soil, h_soil_toe,
                         gamma_soil_dry, gamma_soil_wet, phi, q, surcharge_type,
                         width_surcharge, offset_surcharge, Hwtr, Hwtr_front,
                         gamma_w, kh, gamma_c, c_soil, c_concrete, Es_soil, nu_soil,
                         taper_direction, include_shear_key, shear_key_distance,
                         shear_key_width, shear_key_thickness,
                         enable_pile=False, pile_offset=0.5, L_pile=10.0,
                         diameter_pile=0.4, pile_material='Concrete', Ec=34.8e3, Es=200e3):
    import subprocess
    import sys
    import pickle
    import os

    import tempfile
    scratch_dir = os.path.join(tempfile.gettempdir(), "qrtwall_scratch")
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
        
    pickle_path = os.path.join(scratch_dir, "ssrm_results.pkl")
    pickle_path_escaped = pickle_path.replace("\\", "\\\\")

    E_pile_val = (Es if pile_material == 'Steel' else Ec) * 1000.0

    subprocess_code = f"""
import numpy as np
import pickle
import xslope
from xslope.mesh import build_mesh_from_polygons
from xslope.fem import build_fem_data, solve_ssrm

# Geometry parameters
toe = {toe}
heel = {heel}
bot_wall = {bot_wall}
top_wall = {top_wall}
h_ftg = {h_ftg}
Hw = {Hw}
h_soil = {h_soil}
h_soil_toe = {h_soil_toe}
taper_direction = '{taper_direction}'
include_shear_key = {include_shear_key}
shear_key_distance = {shear_key_distance}
shear_key_width = {shear_key_width}
shear_key_thickness = {shear_key_thickness}
Hwtr = {Hwtr}
Hwtr_front = {Hwtr_front}
gamma_w = {gamma_w}
c_soil = {c_soil}
c_concrete = {c_concrete}
phi = {phi}
gamma_soil_dry = {gamma_soil_dry}
gamma_soil_wet = {gamma_soil_wet}
gamma_c = {gamma_c}
Es_soil = {Es_soil}
nu_soil = {nu_soil}
kh = {kh}

enable_pile = {enable_pile}
pile_offset = {pile_offset}
L_pile = {L_pile}
diameter_pile = {diameter_pile}
E_pile = {E_pile_val}

def _clean_poly(pts):
    clean = []
    for p in pts:
        if not clean or (abs(clean[-1][0] - p[0]) > 1e-6 or abs(clean[-1][1] - p[1]) > 1e-6):
            clean.append(p)
    if len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) < 1e-6 and abs(clean[0][1] - clean[-1][1]) < 1e-6:
        clean.pop()
    return clean

# Geometry coordinates
x0 = 10.0
ftg_len = toe + bot_wall + heel
taper = bot_wall - top_wall

stem_front_ftg_x = x0 + toe + (taper if taper_direction == 'Toe-facing' else 0.0)
stem_back_ftg_x = x0 + toe + bot_wall
stem_front_top_x = x0 + toe + (taper if taper_direction == 'Toe-facing' else 0.0)
stem_back_top_x = x0 + toe + (bot_wall if taper_direction == 'Toe-facing' else top_wall)

wall_coords = [
    (x0, h_ftg),
    (stem_front_ftg_x, h_ftg),
    (stem_front_top_x, h_ftg + Hw),
    (stem_back_top_x, h_ftg + Hw),
    (stem_back_ftg_x, h_ftg),
    (x0 + ftg_len, h_ftg),
    (x0 + ftg_len, 0.0),
]

sk_x1 = x0 + shear_key_distance
sk_x2 = x0 + shear_key_distance + shear_key_width

pile_pts = []
if enable_pile:
    x_pile_toe = x0 + pile_offset
    x_pile_heel = x0 + ftg_len - pile_offset
    pile_pts = [x_pile_heel, x_pile_toe]

if include_shear_key:
    for xp in pile_pts:
        if xp > sk_x2:
            wall_coords.append((xp, 0.0))
    wall_coords.append((sk_x2, 0.0))
    wall_coords.append((sk_x2, -shear_key_thickness))
    for xp in pile_pts:
        if sk_x1 <= xp <= sk_x2:
            wall_coords.append((xp, -shear_key_thickness))
    wall_coords.append((sk_x1, -shear_key_thickness))
    wall_coords.append((sk_x1, 0.0))
    for xp in pile_pts:
        if xp < sk_x1:
            wall_coords.append((xp, 0.0))
else:
    for xp in pile_pts:
        wall_coords.append((xp, 0.0))

wall_coords.append((x0, 0.0))

y_bottom = -max(5.0, h_ftg * 2.0, (L_pile + 5.0) if enable_pile else 0.0)
x_min = 0.0
x_max = x0 + ftg_len + 15.0

soil_coords = [
    (x_min, y_bottom),
    (x_max, y_bottom),
    (x_max, h_ftg + h_soil),
    (stem_back_top_x, h_ftg + h_soil),
    (stem_back_top_x, h_ftg + Hw),
    (stem_back_ftg_x, h_ftg),
    (x0 + ftg_len, h_ftg),
    (x0 + ftg_len, 0.0),
]

if include_shear_key:
    for xp in pile_pts:
        if xp > sk_x2:
            soil_coords.append((xp, 0.0))
    soil_coords.append((sk_x2, 0.0))
    soil_coords.append((sk_x2, -shear_key_thickness))
    for xp in pile_pts:
        if sk_x1 <= xp <= sk_x2:
            soil_coords.append((xp, -shear_key_thickness))
    soil_coords.append((sk_x1, -shear_key_thickness))
    soil_coords.append((sk_x1, 0.0))
    for xp in pile_pts:
        if xp < sk_x1:
            soil_coords.append((xp, 0.0))
else:
    for xp in pile_pts:
        soil_coords.append((xp, 0.0))

soil_coords.extend([
    (x0, 0.0),
    (x0, h_ftg),
    (stem_front_ftg_x, h_ftg),
    (stem_front_ftg_x, h_ftg + min(h_soil_toe, Hw)),
    (x0, h_ftg + h_soil_toe),
    (x_min, h_ftg + h_soil_toe)
])

polygons = [
    {{'coords': _clean_poly(soil_coords), 'mat_id': 0}},
    {{'coords': _clean_poly(wall_coords), 'mat_id': 1}}
]

lines_coords = None
if enable_pile:
    y_pile_toe_top = -shear_key_thickness if (include_shear_key and sk_x1 <= x_pile_toe <= sk_x2) else 0.0
    y_pile_heel_top = -shear_key_thickness if (include_shear_key and sk_x1 <= x_pile_heel <= sk_x2) else 0.0
    pile_line_toe = [(x_pile_toe, y_pile_toe_top), (x_pile_toe, y_pile_toe_top - L_pile)]
    pile_line_heel = [(x_pile_heel, y_pile_heel_top), (x_pile_heel, y_pile_heel_top - L_pile)]
    lines_coords = [pile_line_toe, pile_line_heel]

mesh = build_mesh_from_polygons(polygons=polygons, target_size=1.2, element_type='tri6', lines=lines_coords)

# Piezometric line for groundwater
has_water = (Hwtr > 0 or Hwtr_front > 0)
piezo_line = [(x_min, h_ftg + Hwtr_front), (x0, h_ftg + Hwtr_front), (stem_back_ftg_x, h_ftg + Hwtr), (x_max, h_ftg + Hwtr)]

from shapely.geometry import Polygon
from xslope.fileio import build_ground_surface_from_polygons

soil_poly_clean = _clean_poly(soil_coords)
wall_poly_clean = _clean_poly(wall_coords)
poly_objs = [
    {{'polygon': Polygon(soil_poly_clean), 'mat_id': 0}},
    {{'polygon': Polygon(wall_poly_clean), 'mat_id': 1}}
]
ground_surface, domain_polygon = build_ground_surface_from_polygons(poly_objs)

# Surcharge behind wall
q_val = {q}
dloads = []
if q_val > 0:
    dloads = [
        [
            {{'X': stem_back_top_x, 'Y': h_ftg + h_soil, 'Normal': q_val}},
            {{'X': x_max, 'Y': h_ftg + h_soil, 'Normal': q_val}}
        ]
    ]

slope_data = {{
    'ground_surface': ground_surface,
    'domain_polygon': domain_polygon,
    'polygons': poly_objs,
    'circular': False,
    'materials': [
        {{
            'name': 'Soil',
            'c': c_soil,
            'phi': phi,
            'gamma': gamma_soil_dry,
            'gamma_sat': gamma_soil_wet,
            'E': Es_soil,
            'nu': nu_soil,
            't_cut': 0.0,
            'option': 'mc',
            'd': 0.0,
            'psi': 0.0,
            'u': 'piezo' if has_water else 'none'
        }},
        {{
            'name': 'Concrete Wall',
            'c': c_concrete,
            'phi': 40.0,
            'gamma': gamma_c,
            'E': 40000.0,
            'nu': 0.15,
            't_cut': 0.0,
            'option': 'mc',
            'd': 0.0,
            'psi': 0.0,
            'u': 'none'
        }}
    ],
    'piezo_line': piezo_line if has_water else None,
    'dloads': dloads,
    'gamma_water': gamma_w,
    'k_seismic': kh,
    'tcrack_depth': 0.0,
    'tcrack_water': False,
    'max_depth': y_bottom
}}

# Generate starting circles (matching XSLOPE Studio algorithm)
H_slope = max(1.0, (h_ftg + Hw + h_soil) - (h_ftg + h_soil_toe))
x_toe_surf = x0
y_toe_surf = h_ftg + h_soil_toe
x_crest_surf = stem_back_top_x
y_crest_surf = h_ftg + h_soil

xo1 = x_toe_surf + 0.25 * (stem_back_top_x - x_toe_surf)
yo1 = y_crest_surf + 1.2 * H_slope
depth1 = y_toe_surf - 0.2 * H_slope

xo2 = x_toe_surf + 0.5 * (stem_back_top_x - x_toe_surf)
yo2 = y_crest_surf + 1.5 * H_slope
depth2 = y_toe_surf - 0.5 * H_slope

slope_data['circles'] = [
    {{
        'Xo': xo1,
        'Yo': yo1,
        'Option': 'Depth',
        'Depth': depth1,
        'Xi': 2.0,
        'Yi': 2.0,
        'R': yo1 - depth1
    }},
    {{
        'Xo': xo2,
        'Yo': yo2,
        'Option': 'Depth',
        'Depth': depth2,
        'Xi': 2.0,
        'Yi': 2.0,
        'R': yo2 - depth2
    }}
]

if enable_pile:
    slope_data['pile_lines'] = [
        {{
            'x1': x_pile_toe, 'y1': y_pile_toe_top, 'x2': x_pile_toe, 'y2': y_pile_toe_top - L_pile,
            'E': E_pile,
            'D_pile': diameter_pile,
            'S': 1.5,
            'fixity': 'fixed'
        }},
        {{
            'x1': x_pile_heel, 'y1': y_pile_heel_top, 'x2': x_pile_heel, 'y2': y_pile_heel_top - L_pile,
            'E': E_pile,
            'D_pile': diameter_pile,
            'S': 1.5,
            'fixity': 'fixed'
        }}
    ]

fem_data = build_fem_data(slope_data, mesh=mesh)

# Solve SSRM
res = solve_ssrm(fem_data, F_min=0.3, F_max=3.0, tolerance=0.05, debug_level=0)

result_dict = {{
    "FS": res['FS'],
    "fem_data": fem_data,
    "slope_data": slope_data,
    "last_solution": res.get('last_solution', None)
}}

with open("{pickle_path_escaped}", "wb") as f:
    pickle.dump(result_dict, f)

print("SUCCESS_DONE")
"""

    try:
        res = subprocess.run([sys.executable, '-c', subprocess_code], capture_output=True, text=True)
        if res.returncode == 0 and "SUCCESS_DONE" in res.stdout:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            print("Subprocess failed inside run_2d_ssrm_analysis:", res.stderr)
            return None
    except Exception as e:
        print("Error during run_2d_ssrm_analysis execution:", e)
        return None

@st.cache_data
def get_cached_ssrm_results(toe, heel, bot_wall, top_wall, h_ftg, Hw, h_soil, h_soil_toe,
                            gamma_soil_dry, gamma_soil_wet, phi, q, surcharge_type,
                            width_surcharge, offset_surcharge, Hwtr, Hwtr_front,
                            gamma_w, kh, gamma_c, c_soil, c_concrete, Es_soil, nu_soil,
                            taper_direction, include_shear_key, shear_key_distance,
                            shear_key_width, shear_key_thickness,
                            enable_pile=False, pile_offset=0.5, L_pile=10.0,
                            diameter_pile=0.4, pile_material='Concrete', Ec=34.8e3, Es=200e3):
    return run_2d_ssrm_analysis(
        toe=toe, heel=heel, bot_wall=bot_wall, top_wall=top_wall, h_ftg=h_ftg, Hw=Hw,
        h_soil=h_soil, h_soil_toe=h_soil_toe,
        gamma_soil_dry=gamma_soil_dry, gamma_soil_wet=gamma_soil_wet, phi=phi, q=q,
        surcharge_type=surcharge_type, width_surcharge=width_surcharge, offset_surcharge=offset_surcharge,
        Hwtr=Hwtr, Hwtr_front=Hwtr_front, gamma_w=gamma_w, kh=kh, gamma_c=gamma_c,
        c_soil=c_soil, c_concrete=c_concrete, Es_soil=Es_soil, nu_soil=nu_soil,
        taper_direction=taper_direction, include_shear_key=include_shear_key,
        shear_key_distance=shear_key_distance, shear_key_width=shear_key_width,
        shear_key_thickness=shear_key_thickness,
        enable_pile=enable_pile, pile_offset=pile_offset, L_pile=L_pile,
        diameter_pile=diameter_pile, pile_material=pile_material, Ec=Ec, Es=Es
    )

ssrm_results = None

update_splash(splash_placeholder, 55, "Computing Pile Foundation & Interactions", "Evaluating pile axial/lateral capacity & P-M interaction envelope...")

if enable_pile:
    try:
        from openpile.construct import PileSection, Pile, SoilProfile, Layer, Model
        from openpile.soilmodels import API_sand, API_clay, API_sand_axial, API_clay_axial
        from groundhog.deepfoundations.axialcapacity.skinfriction import API_unit_shaft_friction_sand_rp2geo, API_unit_shaft_friction_clay
        from groundhog.deepfoundations.axialcapacity.endbearing import API_unit_end_bearing_sand_rp2geo, API_unit_end_bearing_clay
        from sectionproperties.pre.geometry import CompoundGeometry
        from sectionproperties.pre.library import circular_section, rectangular_section
        from concreteproperties import (
            Concrete,
            ConcreteLinear,
            ConcreteSection,
            RectangularStressBlock,
            SteelBar,
            SteelElasticPlastic,
        )
        from concreteproperties.pre import add_bar_circular_array, add_bar_rectangular_array
        import matplotlib.path as mpath
        
        # 1. Custom Rectangular Section for OpenPile
        class RectangularPileSection(PileSection):
            top: float
            bottom: float
            width_x: float
            width_y: float

            @property
            def top_elevation(self) -> float:
                return self.top

            @property
            def bottom_elevation(self) -> float:
                return self.bottom

            @property
            def length(self) -> float:
                return self.top - self.bottom

            @property
            def area(self) -> float:
                return self.width_x * self.width_y

            @property
            def footprint(self) -> float:
                return self.width_x * self.width_y

            @property
            def outer_perimeter(self) -> float:
                return 2.0 * (self.width_x + self.width_y)

            @property
            def inner_perimeter(self) -> float:
                return 0.0

            @property
            def entrapped_area(self) -> float:
                return 0.0

            @property
            def second_moment_of_area(self) -> float:
                return self.width_y * (self.width_x ** 3) / 12.0

            @property
            def width(self) -> float:
                return self.width_x

            def get_volume(self, length) -> float:
                return self.area * length

            def get_entrapped_volume(self, length) -> float:
                return 0.0

        # 2. Force split calculations
        d_piles = ftg - 2.0 * pile_offset
        P_toe = Rv_force * (ftg - pile_offset - x_R_loc) / d_piles
        P_heel = Rv_force * (x_R_loc - pile_offset) / d_piles
        
        # Lateral load on piles: use total active driving lateral force + seismic inertia force (neglecting passive resistance for conservative structural design of piles)
        total_active_driving = -sum(Fx.values())
        if kh > 0.0:
            W_conc_seismic = gamma_c * ((top_wall + bot_wall) / 2.0 * Hw + ftg * h_ftg) * t
            W_soil_seismic = (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * (Hwtr + h_ftg)) * ftg * t
            seismic_force = kh * (W_soil_seismic + W_conc_seismic)
        else:
            seismic_force = 0.0
        H_total_piles = max(0.1, total_active_driving + seismic_force)
        
        H_toe = H_total_piles / 2.0
        H_heel = H_total_piles / 2.0

        # 3. Single pile capacities (using groundhog)
        gamma_sub = gamma_pile_soil - gamma_w if (Hwtr_front > 0 or Hwtr > 0) else gamma_pile_soil
        p0_mid = p0_eff + (L_pile / 2.0) * gamma_sub
        p0_tip = p0_eff + L_pile * gamma_sub

        if pile_soil_type == 'Sand (Drained)':
            if phi_pile_soil < 36:
                rd_cat = "Medium dense"
            elif phi_pile_soil < 40:
                rd_cat = "Dense"
            else:
                rd_cat = "Very dense"
            sf_res = API_unit_shaft_friction_sand_rp2geo(api_relativedensity=rd_cat, api_soildescription='Sand', sigma_vo_eff=p0_mid, validate=False)
            f_s_comp = sf_res['f_s_comp_out [kPa]']
            f_s_tens = sf_res['f_s_tens_out [kPa]']
            eb_res = API_unit_end_bearing_sand_rp2geo(api_relativedensity=rd_cat, api_soildescription='Sand', sigma_vo_eff=p0_tip, validate=False)
            q_b = eb_res['q_b_plugged [kPa]']
        else:
            sf_res = API_unit_shaft_friction_clay(undrained_shear_strength=su_pile_soil, sigma_vo_eff=p0_mid, validate=False)
            f_s_comp = sf_res['f_s_comp_out [kPa]']
            f_s_tens = sf_res['f_s_tens_out [kPa]']
            eb_res = API_unit_end_bearing_clay(undrained_shear_strength=su_pile_soil, validate=False)
            q_b = eb_res['q_b_plugged [kPa]']

        if pile_shape == 'Circle':
            A_base = np.pi * (diameter_pile ** 2) / 4.0
            A_shaft = np.pi * diameter_pile * L_pile
        else:
            A_base = width_x_pile * width_y_pile
            A_shaft = 2.0 * (width_x_pile + width_y_pile) * L_pile

        Q_shaft_comp = f_s_comp * A_shaft
        Q_shaft_tens = f_s_tens * A_shaft
        Q_base = q_b * A_base
        Q_comp_ult = Q_shaft_comp + Q_base
        Q_tens_ult = Q_shaft_tens
        Q_comp_allow = Q_comp_ult / FS_pile_axial
        Q_tens_allow = Q_tens_ult / FS_pile_axial

        # Broms' Lateral Capacity
        if pile_soil_type == 'Sand (Drained)':
            Kp = np.tan(np.radians(45.0 + phi_pile_soil / 2.0)) ** 2
            H_ult = 0.5 * gamma_pile_soil * (diameter_pile if pile_shape == 'Circle' else width_x_pile) * (L_pile ** 3) * Kp
        else:
            H_ult = 9.0 * su_pile_soil * (diameter_pile if pile_shape == 'Circle' else width_x_pile) * (L_pile - 1.5 * (diameter_pile if pile_shape == 'Circle' else width_x_pile))
        H_allow = H_ult / FS_pile_lateral

        # 4. OpenPile Lateral & Axial Winkler Analysis
        if pile_shape == 'Circle':
            p_elem = Pile.create_tubular(
                name="Circular Pile",
                top_elevation=0.0,
                bottom_elevation=-L_pile,
                diameter=diameter_pile,
                wt=diameter_pile / 2.0
            )
        else:
            p_elem = Pile(
                name="Rectangular Pile",
                material="Concrete",
                sections=[
                    RectangularPileSection(
                        top=0.0,
                        bottom=-L_pile,
                        width_x=width_x_pile,
                        width_y=width_y_pile
                    )
                ]
            )

        if pile_soil_type == 'Sand (Drained)':
            lat_model = API_sand(phi=phi_pile_soil, kind=pile_loading_type)
            ax_model = API_sand_axial(delta=delta_pile_axial)
        else:
            lat_model = API_clay(Su=[su_pile_soil, su_pile_soil], eps50=eps50_pile_soil, kind=pile_loading_type)
            ax_model = API_clay_axial(Su=[su_pile_soil, su_pile_soil], alpha_limit=alpha_pile_axial)

        sp_elem = SoilProfile(
            name="Pile Soil Profile",
            top_elevation=0.0,
            water_line=0.0 if (Hwtr_front > 0 or Hwtr > 0) else -L_pile,
            layers=[
                Layer(
                    name="layer1",
                    top=0.0,
                    bottom=-L_pile,
                    weight=gamma_pile_soil,
                    lateral_model=lat_model,
                    axial_model=ax_model
                )
            ]
        )

        # Winkler solve for toe
        M_toe_model = Model(name="Toe Pile Model", pile=p_elem, soil=sp_elem, element_type="EulerBernoulli")
        M_toe_model.set_support(elevation=-L_pile, Tz=True)
        M_toe_model.set_pointload(elevation=0.0, Py=H_toe, Pz=-P_toe)
        res_toe = M_toe_model.solve()

        # Winkler solve for heel
        M_heel_model = Model(name="Heel Pile Model", pile=p_elem, soil=sp_elem, element_type="EulerBernoulli")
        M_heel_model.set_support(elevation=-L_pile, Tz=True)
        M_heel_model.set_pointload(elevation=0.0, Py=H_heel, Pz=-P_heel)
        res_heel = M_heel_model.solve()

        # Extract winkler results
        V_max_toe = max(abs(res_toe.forces['V [kN]']))
        M_max_toe = max(abs(res_toe.forces['M [kNm]']))
        V_max_heel = max(abs(res_heel.forces['V [kN]']))
        M_max_heel = max(abs(res_heel.forces['M [kNm]']))

        # Extract axial pile head settlements (elevation = 0.0) in mm
        try:
            settlement_toe_pile = abs(res_toe.settlement.loc[res_toe.settlement['Elevation [m]'] == 0.0, 'Settlement [m]'].values[0]) * 1000.0
            settlement_heel_pile = abs(res_heel.settlement.loc[res_heel.settlement['Elevation [m]'] == 0.0, 'Settlement [m]'].values[0]) * 1000.0
        except Exception:
            settlement_toe_pile = 0.0
            settlement_heel_pile = 0.0

        # Allowable pile settlement = 2% of diameter
        pile_dia_val = diameter_pile if pile_shape == 'Circle' else width_x_pile
        allowable_pile_settlement_mm = 0.02 * pile_dia_val * 1000.0
        pile_pass_toe_settlement = settlement_toe_pile <= allowable_pile_settlement_mm
        pile_pass_heel_settlement = settlement_heel_pile <= allowable_pile_settlement_mm

        # 5. concrete-properties Interaction Diagram
        material_concrete_pile = Concrete(
            name="Pile Concrete",
            density=2.4e-6,
            stress_strain_profile=ConcreteLinear(elastic_modulus=Ec),
            ultimate_stress_strain_profile=RectangularStressBlock(
                compressive_strength=fc,
                alpha=0.85,
                gamma=0.85,
                ultimate_strain=0.003
            ),
            flexural_tensile_strength=3.0,
            colour="lightgrey"
        )
        material_steel_pile = SteelBar(
            name="Pile Rebar",
            density=7.85e-6,
            stress_strain_profile=SteelElasticPlastic(
                yield_strength=fy,
                elastic_modulus=Es,
                fracture_strain=0.05
            ),
            colour="grey"
        )

        if pile_material == 'Concrete':
            if pile_shape == 'Circle':
                geom_pile = circular_section(d=diameter_pile * 1000.0, n=32, material=material_concrete_pile)
                rebar_area = np.pi * (rebar_dia_pile ** 2) / 4.0
                geom_pile = add_bar_circular_array(
                    geometry=geom_pile,
                    area=rebar_area,
                    material=material_steel_pile,
                    n_bar=int(n_rebar_pile),
                    r_array=diameter_pile * 1000.0 / 2.0 - cover_pile,
                    ctr=(0.0, 0.0)
                )
            else:
                geom_pile = rectangular_section(d=width_x_pile * 1000.0, b=width_y_pile * 1000.0, material=material_concrete_pile)
                rebar_area = np.pi * (rebar_dia_pile ** 2) / 4.0
                geom_pile = add_bar_rectangular_array(
                    geometry=geom_pile,
                    area=rebar_area,
                    material=material_steel_pile,
                    n_x=2,
                    x_s=width_y_pile * 1000.0 - 2.0 * cover_pile,
                    n_y=2,
                    y_s=width_x_pile * 1000.0 - 2.0 * cover_pile,
                    anchor=(cover_pile, cover_pile)
                )
        else:
            # Steel solid section model
            steel_concrete = Concrete(
                name="Solid Steel",
                density=7.85e-6,
                stress_strain_profile=ConcreteLinear(elastic_modulus=Es),
                ultimate_stress_strain_profile=RectangularStressBlock(
                    compressive_strength=fy,
                    alpha=1.0,
                    gamma=1.0,
                    ultimate_strain=0.05
                ),
                flexural_tensile_strength=fy,
                colour="grey"
            )
            if pile_shape == 'Circle':
                geom_pile = circular_section(d=diameter_pile * 1000.0, n=32, material=steel_concrete)
            else:
                geom_pile = rectangular_section(d=width_x_pile * 1000.0, b=width_y_pile * 1000.0, material=steel_concrete)

        if not isinstance(geom_pile, CompoundGeometry):
            geom_pile = CompoundGeometry([geom_pile])
        sec_pile = ConcreteSection(geom_pile)

        # Generate Pile section geometry plot
        plt.close('all')
        sec_pile.plot_section()
        fig_pile_sec = plt.gcf()
        fig_pile_sec.set_size_inches(5, 4)
        try:
            ax = fig_pile_sec.axes[0]
            ax.set_title("Pile Cross Section Geometry", fontsize=10)
        except Exception:
            pass
        fig_pile_sec.tight_layout()
        try:
            import os
            os.makedirs("report_temp", exist_ok=True)
            fig_pile_sec.savefig("report_temp/pile_section.png", dpi=150, bbox_inches='tight')
        except Exception as e_save:
            print("Error saving pile_section.png:", e_save)
        plt.close(fig_pile_sec)

        mi_res = sec_pile.moment_interaction_diagram(control_points=[('kappa0', 0.0), ('d_n', 1e-6)], progress_bar=False)
        pos_m = [r.m_x / 1.0e6 for r in mi_res.results]
        pos_n = [r.n / 1000.0 for r in mi_res.results]

        curve_m = pos_m + [-x for x in pos_m[::-1]]
        curve_n = pos_n + pos_n[::-1]
        polygon_points = list(zip(curve_m, curve_n))
        path_pile = mpath.Path(polygon_points)

        # Verification Checks
        pile_pass_toe_axial = P_toe <= Q_comp_allow if P_toe >= 0 else abs(P_toe) <= Q_tens_allow
        pile_pass_toe_lateral = V_max_toe <= H_allow
        pile_pass_toe_interaction = path_pile.contains_point((M_max_toe, P_toe))

        pile_pass_heel_axial = P_heel <= Q_comp_allow if P_heel >= 0 else abs(P_heel) <= Q_tens_allow
        pile_pass_heel_lateral = V_max_heel <= H_allow
        pile_pass_heel_interaction = path_pile.contains_point((M_max_heel, P_heel))

    except Exception as e:
        st.error(f"Error during Pile Analysis: {e}")
        import traceback
        st.code(traceback.format_exc())

# =========================================================
# GLOBAL STABILITY ANALYSIS (always runs)
# =========================================================
try:
    update_splash(splash_placeholder, 80, "Solving XSLOPE 2D SSRM Slope Stability", "Meshing tri6 quadratic elements & computing Mohr-Coulomb viscoplastic failure surface...")
    # 2D SSRM Slope Stability Analysis
    ssrm_results = get_cached_ssrm_results(
        toe=toe,
        heel=heel,
        bot_wall=bot_wall,
        top_wall=top_wall,
        h_ftg=h_ftg,
        Hw=Hw,
        h_soil=h_soil,
        h_soil_toe=h_soil_toe,
        gamma_soil_dry=gamma_soil_dry,
        gamma_soil_wet=gamma_soil_wet,
        phi=phi,
        q=q,
        surcharge_type=surcharge_type,
        width_surcharge=width_surcharge,
        offset_surcharge=offset_surcharge,
        Hwtr=Hwtr,
        Hwtr_front=Hwtr_front,
        gamma_w=gamma_w,
        kh=PGA * FPGA,
        gamma_c=gamma_c,
        c_soil=c_soil,
        c_concrete=c_concrete,
        Es_soil=Es_soil,
        nu_soil=nu_soil,
        taper_direction=taper_direction,
        include_shear_key=include_shear_key,
        shear_key_distance=shear_key_distance if include_shear_key else 0.0,
        shear_key_width=shear_key_width if include_shear_key else 0.0,
        shear_key_thickness=shear_key_thickness if include_shear_key else 0.0,
        enable_pile=enable_pile,
        pile_offset=pile_offset,
        L_pile=L_pile,
        diameter_pile=diameter_pile,
        pile_material=pile_material,
        Ec=Ec,
        Es=Es
    )

except Exception as e:
    st.error(f"Error during Global Stability Analysis: {e}")
    import traceback
    st.code(traceback.format_exc())

# =========================================================
# DASHBOARD UI RENDER
# =========================================================

def status_span(ok):
    if ok:
        return "<span style='background-color:#28a745; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px;'>PASS</span>"
    return "<span style='background-color:#dc3545; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px;'>FAIL</span>"


# 1. Title Banner
st.markdown("""
<div class="dashboard-header">
    <h1>🧱 RT Wall Cantilever</h1>
    <p>Preliminary Cantilever Retaining Wall Design tool</p>
</div>
""", unsafe_allow_html=True)

# 2. KPI Summary Row
if enable_bearing:
    kpi_cols = st.columns(6)
else:
    kpi_cols = st.columns(5)

# Determine concrete checking status
stem_pass = M_actual_stem <= M_u_stem_allow if M_u_stem_allow > 0 else False
toe_pass = M_actual_toe <= M_u_ftg_sag_allow if M_u_ftg_sag_allow > 0 else False
heel_pass = M_actual_heel <= M_u_ftg_hog_allow if M_u_ftg_hog_allow > 0 else False
concrete_pass = stem_pass and toe_pass and heel_pass

with kpi_cols[0]:
    with st.container(border=True):
        if FS_ot_global is not None:
            status_ot = "PASS (Min 1.5)" if FS_ot_global >= 1.5 else "FAIL (Min 1.5)"
            delta_ot = "normal" if FS_ot_global >= 1.5 else "inverse"
            st.metric(label="Global Overturning FS", value=f"{FS_ot_global:.3f}", delta=status_ot, delta_color=delta_ot)
        else:
            st.metric(label="Global Overturning FS", value="N/A")

with kpi_cols[1]:
    with st.container(border=True):
        if FS_slide_global is not None:
            status_slide = "PASS (Min 1.5)" if FS_slide_global >= 1.5 else "FAIL (Min 1.5)"
            delta_slide = "normal" if FS_slide_global >= 1.5 else "inverse"
            st.metric(label="Global Sliding FS", value=f"{FS_slide_global:.3f}", delta=status_slide, delta_color=delta_slide)
        else:
            st.metric(label="Global Sliding FS", value="N/A")

with kpi_cols[2]:
    with st.container(border=True):
        if FS_ot_stem is not None:
            status_stem = "PASS (Min 1.5)" if FS_ot_stem >= 1.5 else "FAIL (Min 1.5)"
            delta_stem = "normal" if FS_ot_stem >= 1.5 else "inverse"
            st.metric(label="Stem Junction Overturning FS", value=f"{FS_ot_stem:.3f}", delta=status_stem, delta_color=delta_stem)
        else:
            st.metric(label="Stem Junction Overturning FS", value="N/A")

if enable_bearing:
    with kpi_cols[3]:
        with st.container(border=True):
            status_bearing = f"PASS (Min {FS_bearing:.2f})" if bearing_pass else f"FAIL (Min {FS_bearing:.2f})"
            delta_bearing = "normal" if bearing_pass else "inverse"
            bearing_fs_str = f"{FS_actual_bearing:.3f}" if FS_actual_bearing != float('inf') else "∞"
            st.metric(label="Bearing Capacity FS", value=bearing_fs_str, delta=status_bearing, delta_color=delta_bearing)
    disp_col_idx = 4
    concrete_col_idx = 5
else:
    disp_col_idx = 3
    concrete_col_idx = 4

with kpi_cols[disp_col_idx]:
    with st.container(border=True):
        st.metric(label="Max Nodal Disp (X / Y)", value=f"{max_disp_x*1000:.2f} / {max_disp_y*1000:.2f} mm", delta="FEM Peak", delta_color="off")

with kpi_cols[concrete_col_idx]:
    with st.container(border=True):
        st.metric(label="Concrete Strength", 
                  value="PASS" if concrete_pass else "FAIL", 
                  delta="Design Safe" if concrete_pass else "Overstressed",
                  delta_color="normal" if concrete_pass else "inverse")

# 2b. Settlement KPI Row
st.markdown("### 📏 Settlement & Deformation KPIs (SNI 8460:2017)")
settlement_cols = st.columns(4)

with settlement_cols[0]:
    with st.container(border=True):
        status_diff = "PASS (≤50mm)" if diff_settlement_pass else "FAIL (≤50mm)"
        delta_color_diff = "normal" if diff_settlement_pass else "inverse"
        st.metric(label="FEM Diff. Settlement", value=f"{diff_settlement_mm:.2f} mm", delta=status_diff, delta_color=delta_color_diff)

with settlement_cols[1]:
    with st.container(border=True):
        st.metric(label="FEM Rotation", value=f"{rotation_deg:.4f}°", delta="Toe vs Heel", delta_color="off")

if enable_bearing:
    with settlement_cols[2]:
        with st.container(border=True):
            status_consol = f"PASS (≤{consol_allow_mm:.1f}mm)" if consol_pass else f"FAIL (≤{consol_allow_mm:.1f}mm)"
            delta_color_consol = "normal" if consol_pass else "inverse"
            st.metric(label="Consol. Settlement", value=f"{consol_settlement_mm:.2f} mm", delta=status_consol, delta_color=delta_color_consol)
else:
    with settlement_cols[2]:
        with st.container(border=True):
            st.metric(label="Consol. Settlement", value="N/A", delta="Not Enabled", delta_color="off")

if enable_pile:
    with settlement_cols[3]:
        with st.container(border=True):
            max_pile_settle = max(settlement_toe_pile, settlement_heel_pile)
            pile_settle_pass = max_pile_settle <= allowable_pile_settlement_mm
            status_pile_settle = f"PASS (≤{allowable_pile_settlement_mm:.1f}mm)" if pile_settle_pass else f"FAIL (≤{allowable_pile_settlement_mm:.1f}mm)"
            delta_color_pile_settle = "normal" if pile_settle_pass else "inverse"
            st.metric(label="Max Pile Settlement", value=f"{max_pile_settle:.2f} mm", delta=status_pile_settle, delta_color=delta_color_pile_settle)
else:
    with settlement_cols[3]:
        with st.container(border=True):
            st.metric(label="Max Pile Settlement", value="N/A", delta="Not Enabled", delta_color="off")

# 3. Two-Column Dashboard Content Grid
col_left, col_right = st.columns([1, 1])

# --- LEFT COLUMN: MODEL & LOADS ---
with col_left:
    
    # Card A: Model Geometry & FE Mesh
    with st.container(border=True):
        st.subheader("📐 Model Geometry & FEM Mesh")
        st.markdown('<div class="section-desc">Dimensioned drawing of retaining wall components, soil blocks, water levels, and centroids.</div>', unsafe_allow_html=True)
        
        # Draw the model dynamically using drawsvg
        try:
            mult = 1000
            # Dynamically compute drawing height to prevent clipping when soil or water is high
            max_height = max(Hw, h_soil, h_soil_toe, Hwtr, Hwtr_front) + h_ftg
            if include_shear_key:
                max_height += shear_key_thickness
            d = draw.Drawing((ftg + 1.8)*mult, (max_height + 2.8)*mult, origin='bottom-left')
            if include_shear_key:
                min_y = -(max_height + 2.8) * mult
                max_y = (shear_key_thickness + 0.3) * mult
                d.view_box = (0, min_y, (ftg + 1.8)*mult, max_y - min_y)
            
            # Draw soil - Dry soil (above water table)
            if 'soil_dry_poly' in globals() and soil_dry_poly:
                pts_dry = []
                for px, py in soil_dry_poly:
                    pts_dry.extend([px*mult, py*mult])
                d.append(draw.Lines(*pts_dry, close=True, fill='#5BC2A5', stroke='black', stroke_width=3))

            # Saturated soil (below water table)
            if 'soil_wet_poly' in globals() and soil_wet_poly:
                pts_wet = []
                for px, py in soil_wet_poly:
                    pts_wet.extend([px*mult, py*mult])
                d.append(draw.Lines(*pts_wet, close=True, fill='#A6F527', stroke='black', stroke_width=3))

            # Soil in front of toe
            if h_soil_toe > 0:
                d.append(draw.Lines(0, -h_ftg*mult,
                                        toe*mult, -h_ftg*mult,
                                        toe*mult, -(h_ftg + h_soil_toe)*mult,
                                        0, -(h_ftg + h_soil_toe)*mult,
                                        close=True,
                                fill='#5BC2A5',
                                stroke='black',
                                stroke_width=3))

            # Water level lines
            # Front water level
            if Hwtr_front > 0:
                d.append(draw.Line(0, -(h_ftg + Hwtr_front)*mult,
                                        toe*mult, -(h_ftg + Hwtr_front)*mult,
                                        stroke='blue', stroke_width=15, stroke_dasharray='50,50'))

            # Back water level
            if Hwtr > 0:
                d.append(draw.Line((toe+bot_wall)*mult, -(h_ftg + Hwtr)*mult,
                                        ftg*mult, -(h_ftg + Hwtr)*mult,
                                        stroke='blue', stroke_width=15, stroke_dasharray='50,50'))

            # Draw Retaining wall
            wall_points_d = [
                0.0, 0.0,
                0.0, -h_ftg*mult,
                toe*mult, -h_ftg*mult
            ]
            for px, py in stem_pts:
                wall_points_d.extend([px*mult, py*mult])
            wall_points_d.extend([
                ftg*mult, -h_ftg*mult,
                ftg*mult, 0.0
            ])
            d.append(draw.Lines(*wall_points_d,
                                    close=True,
                            fill='#eeee00',
                            stroke='black',
                            stroke_width=50))

            # Draw Shear Key (below footing base)
            if include_shear_key:
                sk_x1 = shear_key_distance * mult
                sk_x2 = (shear_key_distance + shear_key_width) * mult
                sk_y1 = 0.0   # footing base level
                sk_y2 = shear_key_thickness * mult  # extends downward (positive in drawing coords)
                d.append(draw.Lines(sk_x1, sk_y1,
                                    sk_x2, sk_y1,
                                    sk_x2, sk_y2,
                                    sk_x1, sk_y2,
                                    close=True,
                                    fill='#FFD700',
                                    stroke='black',
                                    stroke_width=30))
                # Label
                d.append(draw.Text('Shear Key', 0.15*mult,
                                   (sk_x1 + sk_x2) / 2, sk_y2 + 0.12*mult,
                                   fill='black', text_anchor='middle', font_weight='bold'))

            # Vertical dimensions
            x_dim_v1 = (ftg + 0.3) * mult
            x_dim_v2 = (ftg + 0.9) * mult

            d.append(draw.Lines(ftg*mult, 0, x_dim_v2, 0, stroke='gray', stroke_width=10))
            d.append(draw.Lines(ftg*mult, -h_ftg*mult, x_dim_v1, -h_ftg*mult, stroke='gray', stroke_width=10))
            d.append(draw.Lines((toe+bot_wall)*mult, -(h_ftg+Hw)*mult, x_dim_v2, -(h_ftg+Hw)*mult, stroke='gray', stroke_width=10))

            # 1. Footing thickness (vertical)
            d.append(draw.Lines(x_dim_v1, 0, x_dim_v1, -h_ftg*mult, stroke='black', stroke_width=20))
            d.append(draw.Circle(x_dim_v1, 0, 0.06*mult))
            d.append(draw.Circle(x_dim_v1, -h_ftg*mult, 0.06*mult))
            d.append(draw.Text(f"{h_ftg} m", 0.2*mult, x_dim_v1 + (0.05*mult), -h_ftg*mult/2, fill='black'))

            # 2. Wall height (vertical)
            d.append(draw.Lines(x_dim_v1, -h_ftg*mult, x_dim_v1, -(h_ftg+Hw)*mult, stroke='black', stroke_width=20))
            d.append(draw.Circle(x_dim_v1, -h_ftg*mult, 0.06*mult))
            d.append(draw.Circle(x_dim_v1, -(h_ftg+Hw)*mult, 0.06*mult))
            d.append(draw.Text(f"{Hw} m", 0.2*mult, x_dim_v1 + (0.05*mult), -(h_ftg + Hw/2)*mult, fill='black'))

            # Total height (vertical)
            d.append(draw.Lines(x_dim_v2, 0, x_dim_v2, -(h_ftg+Hw)*mult, stroke='black', stroke_width=30))
            d.append(draw.Circle(x_dim_v2, 0, 0.08*mult))
            d.append(draw.Circle(x_dim_v2, -(h_ftg+Hw)*mult, 0.08*mult))
            d.append(draw.Text(f"{Hw+h_ftg} m", 0.25*mult, x_dim_v2 + (0.1*mult), -(h_ftg+Hw)*mult/2, fill='black'))

            # Horizontal dimensions
            y_dim_h1 = -(h_ftg + Hw + 0.3) * mult
            y_dim_h2 = -(h_ftg + Hw + 0.8) * mult
            y_dim_h3 = -(h_ftg + Hw + 1.4) * mult

            d.append(draw.Lines(0, 0, 0, y_dim_h3, stroke='gray', stroke_width=10))
            d.append(draw.Lines(toe*mult, -h_ftg*mult, toe*mult, y_dim_h2, stroke='gray', stroke_width=10))
            d.append(draw.Lines(top_left_stem_x*mult, -(h_ftg+Hw)*mult, top_left_stem_x*mult, y_dim_h1, stroke='gray', stroke_width=10))
            d.append(draw.Lines(top_right_stem_x*mult, -(h_ftg+Hw)*mult, top_right_stem_x*mult, y_dim_h2, stroke='gray', stroke_width=10))
            d.append(draw.Lines(ftg*mult, 0, ftg*mult, y_dim_h3, stroke='gray', stroke_width=10))

            # 3. Top wall thickness
            x_top_1 = top_left_stem_x * mult
            x_top_2 = top_right_stem_x * mult
            d.append(draw.Lines(x_top_1, y_dim_h1, x_top_2, y_dim_h1, stroke='black', stroke_width=20))
            d.append(draw.Circle(x_top_1, y_dim_h1, 0.06*mult))
            d.append(draw.Circle(x_top_2, y_dim_h1, 0.06*mult))
            d.append(draw.Text(f"{top_wall} m", 0.2*mult, (x_top_1+x_top_2)/2, y_dim_h1 - (0.05*mult), fill='black', text_anchor='middle'))

            # 4. Toe length
            x_toe_1 = 0
            x_toe_2 = toe * mult
            d.append(draw.Lines(x_toe_1, y_dim_h2, x_toe_2, y_dim_h2, stroke='black', stroke_width=20))
            d.append(draw.Circle(x_toe_1, y_dim_h2, 0.06*mult))
            d.append(draw.Circle(x_toe_2, y_dim_h2, 0.06*mult))
            d.append(draw.Text(f"{toe} m", 0.2*mult, (x_toe_1+x_toe_2)/2, y_dim_h2 - (0.05*mult), fill='black', text_anchor='middle'))

            # 5. Bottom wall thickness
            x_bot_1 = toe * mult
            x_bot_2 = (toe + bot_wall) * mult
            d.append(draw.Lines(x_bot_1, y_dim_h2, x_bot_2, y_dim_h2, stroke='black', stroke_width=20))
            d.append(draw.Circle(x_bot_1, y_dim_h2, 0.06*mult))
            d.append(draw.Circle(x_bot_2, y_dim_h2, 0.06*mult))
            d.append(draw.Text(f"{bot_wall} m", 0.2*mult, (x_bot_1+x_bot_2)/2, y_dim_h2 - (0.05*mult), fill='black', text_anchor='middle'))

            # 6. Heel length
            x_heel_1 = (toe + bot_wall) * mult
            x_heel_2 = ftg * mult
            d.append(draw.Lines(x_heel_1, y_dim_h2, x_heel_2, y_dim_h2, stroke='black', stroke_width=20))
            d.append(draw.Circle(x_heel_1, y_dim_h2, 0.06*mult))
            d.append(draw.Circle(x_heel_2, y_dim_h2, 0.06*mult))
            d.append(draw.Text(f"{heel} m", 0.2*mult, (x_heel_1+x_heel_2)/2, y_dim_h2 - (0.05*mult), fill='black', text_anchor='middle'))

            # Total footing length
            d.append(draw.Lines(0, y_dim_h3, ftg*mult, y_dim_h3, stroke='black', stroke_width=30))
            d.append(draw.Circle(0, y_dim_h3, 0.08*mult))
            d.append(draw.Circle(ftg*mult, y_dim_h3, 0.08*mult))
            d.append(draw.Text(f"{ftg} m", 0.25*mult, (ftg*mult)/2, y_dim_h3 - (0.1*mult), fill='black', text_anchor='middle'))

            d.set_render_size(600, 400)

            # add centroid markers (red) for base, stem, and soil
            r = 0.03 * mult
            def add_x_lines(poly):
                if poly and len(poly) > 0:
                    tl = min(poly, key=lambda p: p[0] + p[1])
                    br = max(poly, key=lambda p: p[0] + p[1])
                    tr = max(poly, key=lambda p: p[0] - p[1])
                    bl = min(poly, key=lambda p: p[0] - p[1])
                    d.append(draw.Lines(tl[0] * mult, tl[1] * mult, br[0] * mult, br[1] * mult, stroke='black', stroke_dasharray='50,50', stroke_width=10))
                    d.append(draw.Lines(bl[0] * mult, bl[1] * mult, tr[0] * mult, tr[1] * mult, stroke='black', stroke_dasharray='50,50', stroke_width=10))

            if stem_centroid[0] is not None:
                sx, sy = stem_centroid
                add_x_lines(poly=stem_poly)
                d.append(draw.Circle(sx * mult, sy * mult, r * 1.4, fill='red', stroke='black', stroke_width=2))
            if base_centroid[0] is not None:
                bx, by = base_centroid
                add_x_lines(poly=base_poly)
                d.append(draw.Circle(bx * mult, by * mult, r * 1.4, fill='red', stroke='black', stroke_width=2))
            if 'soil_dry_centroid' in locals() and soil_dry_centroid[0] is not None:
                sd_x, sd_y = soil_dry_centroid
                add_x_lines(poly=soil_dry_poly)
                d.append(draw.Circle(sd_x * mult, sd_y * mult, r * 1.4, fill='red', stroke='black', stroke_width=2))
            if 'soil_wet_centroid' in locals() and soil_wet_centroid[0] is not None:
                sw_x, sw_y = soil_wet_centroid
                add_x_lines(poly=soil_wet_poly)
                d.append(draw.Circle(sw_x * mult, sw_y * mult, r * 1.4, fill='red', stroke='black', stroke_width=2))
                
            d.save_png('Rtwall.png')
        except Exception as e:
            st.error(f"Error drawing SVG model: {e}")

        # Display the SVG directly
        try:
            st.image(d.as_svg(), use_container_width=True)
        except Exception as e_svg:
            st.error(f"Error displaying SVG directly: {e_svg}")
            # Fallback to static png image
            st.image('Rtwall.png', use_container_width=True)
        
        # Expandable OpenSees Node Plot
        with st.expander("🔍 View OpenSees Finite Element Mesh Plot"):
            plt.close('all')
            opsv.plot_model("nodes", "elements")
            fig_model = plt.gcf()
            fig_model.set_size_inches(6, 4.5)
            if getattr(fig_model, 'axes', None):
                ax = fig_model.axes[0]
                try:
                    ax.relim()
                    ax.autoscale_view()
                except Exception:
                    pass
            fig_model.tight_layout()
            st.pyplot(fig_model)
            plt.close(fig_model)

    # Card A.2: SNI Geotechnical Geometry Verification
    with st.container(border=True):
        st.subheader("🛡️ SNI Geotechnical Geometry Verification Check")
        st.markdown('<div class="section-desc">Verification of typical dimensions based on Indonesian Geotechnical Standard (SNI Perencanaan Geoteknik - Gambar 35).</div>', unsafe_allow_html=True)

        try:
            # 1. Proportions checks
            H_tot = Hw + h_ftg
            top_wall_ok = top_wall >= 0.3
            bot_wall_ok = bot_wall >= 0.1 * H_tot
            slope_ok = (taper / Hw) >= (1.0 / 48.0) if Hw > 0 else True
            ftg_ok = (0.4 * H_tot) <= ftg <= (0.7 * H_tot)
            h_ftg_ok = (H_tot / 12.0) <= h_ftg <= (H_tot / 10.0)
            toe_ok = toe >= (ftg / 3.0)
            
            # 2. Draw verification model using drawsvg
            mult = 1000
            y_shift_sni = 1.5 * mult
            x_shift_sni = 0.8 * mult
            
            d_sni = draw.Drawing((ftg + 2.2)*mult, (H_tot + 1.8)*mult, origin='bottom-left')
            
            # Helper to draw a CAD-style dimension label directly on or near the line
            def draw_dim_cad_sni(d, x1, y1, x2, y2, text_label, value_str, req_str, is_ok, is_vertical=False, text_pos='right_or_top', text_y_override=None):
                color = '#27AE60' if is_ok else '#C0392B'
                status_str = "[PASS]" if is_ok else "[FAIL]"
                
                # 1. Main dimension line
                d.append(draw.Line(x1, y1, x2, y2, stroke=color, stroke_width=10))
                
                # 2. Tick lines at ends
                tick = 0.08 * mult
                if is_vertical:
                    d.append(draw.Line(x1 - tick, y1, x1 + tick, y1, stroke=color, stroke_width=10))
                    d.append(draw.Line(x2 - tick, y2, x2 + tick, y2, stroke=color, stroke_width=10))
                    
                    # Aligned text label to the right
                    tx = x1 + 0.15 * mult
                    ty = (y1 + y2) / 2
                    d.append(draw.Text(f"{text_label} = {value_str}", 0.18*mult, tx, ty + 0.1*mult, fill=color, font_weight='bold'))
                    d.append(draw.Text(f"Req: {req_str} {status_str}", 0.14*mult, tx, ty - 0.1*mult, fill=color))
                else:
                    d.append(draw.Line(x1, y1 - tick, x1, y1 + tick, stroke=color, stroke_width=10))
                    d.append(draw.Line(x2, y2 - tick, x2, y2 + tick, stroke=color, stroke_width=10))
                    
                    # Aligned text label above or below
                    tx = (x1 + x2) / 2
                    if text_y_override is not None:
                        ty = text_y_override
                        d.append(draw.Text(f"{text_label} = {value_str}", 0.18*mult, tx, ty, fill=color, font_weight='bold', text_anchor='middle'))
                        d.append(draw.Text(f"Req: {req_str} {status_str}", 0.14*mult, tx, ty + 0.20*mult, fill=color, text_anchor='middle'))
                    else:
                        if text_pos == 'top':
                            ty = y1 + 0.1 * mult
                            d.append(draw.Text(f"{text_label} = {value_str}", 0.18*mult, tx, ty + 0.16*mult, fill=color, font_weight='bold', text_anchor='middle'))
                            d.append(draw.Text(f"Req: {req_str} {status_str}", 0.14*mult, tx, ty, fill=color, text_anchor='middle'))
                        else:
                            ty = y1 - 0.18 * mult
                            d.append(draw.Text(f"{text_label} = {value_str}", 0.18*mult, tx, ty, fill=color, font_weight='bold', text_anchor='middle'))
                            d.append(draw.Text(f"Req: {req_str} {status_str}", 0.14*mult, tx, ty - 0.15*mult, fill=color, text_anchor='middle'))

            # Draw retaining wall structure (yellow)
            wall_points_sni = [
                x_shift_sni,  -y_shift_sni,
                x_shift_sni,  -h_ftg*mult - y_shift_sni,
                toe*mult + x_shift_sni, -h_ftg*mult - y_shift_sni
            ]
            for px, py in stem_pts:
                wall_points_sni.extend([px*mult + x_shift_sni, py*mult - y_shift_sni])
            wall_points_sni.extend([
                ftg*mult + x_shift_sni, -h_ftg*mult - y_shift_sni,
                ftg*mult + x_shift_sni, -y_shift_sni
            ])
            d_sni.append(draw.Lines(*wall_points_sni,
                                    close=True,
                                    fill='#eeee00',
                                    stroke='black',
                                    stroke_width=25))

            # 1. Total Height H (vertical on right)
            draw_dim_cad_sni(d_sni, (ftg + 0.15)*mult + x_shift_sni, -y_shift_sni,
                                     (ftg + 0.15)*mult + x_shift_sni, -H_tot*mult - y_shift_sni,
                                     "Height H", f"{H_tot:.2f} m", "Reference", True, is_vertical=True)

            # 2. Footing Thickness D (vertical on right)
            draw_dim_cad_sni(d_sni, (ftg + 0.85)*mult + x_shift_sni, -y_shift_sni,
                                     (ftg + 0.85)*mult + x_shift_sni, -h_ftg*mult - y_shift_sni,
                                     "Thickness D", f"{h_ftg:.2f} m", f"{H_tot/12:.2f}~{H_tot/10:.2f} m", h_ftg_ok, is_vertical=True)

            # Aligned Y coordinate for toe length, Width B, and base wall dimension lines (below the bottom of footing -1500)
            y_dim_bottom = -y_shift_sni + 0.15*mult

            # 3. Footing Width B (at bottom) - Text level 3 (Width B text below base wall text)
            draw_dim_cad_sni(d_sni, x_shift_sni, y_dim_bottom,
                                     ftg*mult + x_shift_sni, y_dim_bottom,
                                     "Width B", f"{ftg:.2f} m", f"{0.4*H_tot:.2f}~{0.7*H_tot:.2f} m", ftg_ok, is_vertical=False, text_y_override=y_dim_bottom + 1.0*mult)

            # 4. Top Wall thickness (above stem top)
            draw_dim_cad_sni(d_sni, top_left_stem_x*mult + x_shift_sni, -(h_ftg + Hw + 0.15)*mult - y_shift_sni,
                                     top_right_stem_x*mult + x_shift_sni, -(h_ftg + Hw + 0.15)*mult - y_shift_sni,
                                     "Top Wall", f"{top_wall:.2f} m", ">= 0.30 m", top_wall_ok, is_vertical=False, text_pos='top')

            # 5. Bottom Wall thickness (Base Wall) - Text level 2 (base wall text below toe length text)
            draw_dim_cad_sni(d_sni, toe*mult + x_shift_sni, y_dim_bottom,
                                     (toe + bot_wall)*mult + x_shift_sni, y_dim_bottom,
                                     "Base Wall", f"{bot_wall:.2f} m", f">={0.1*H_tot:.2f} m", bot_wall_ok, is_vertical=False, text_y_override=y_dim_bottom + 0.5*mult)

            # 6. Toe length - Text level 1 (first toe length text closest to dimension line)
            draw_dim_cad_sni(d_sni, x_shift_sni, y_dim_bottom,
                                     toe*mult + x_shift_sni, y_dim_bottom,
                                     "Toe Length", f"{toe:.2f} m", f">={ftg/3:.2f} m", toe_ok, is_vertical=False, text_y_override=y_dim_bottom + 0.0*mult)

            # 7. Front Slope Batter
            if taper_direction == 'Toe-facing':
                slope_val = taper / Hw if Hw > 0 else 0
                slope_str = f"1 : {1.0/slope_val:.1f}" if slope_val > 0 else "Vertical"
                slope_req = ">= 1:48"
            else:
                slope_val = 0.0
                slope_str = "Vertical"
                slope_req = "N/A"
            draw_dim_cad_sni(d_sni, batter_mid_x*mult + x_shift_sni - 0.15*mult, -(h_ftg + Hw/2)*mult - y_shift_sni,
                                     batter_mid_x*mult + x_shift_sni + 0.15*mult, -(h_ftg + Hw/2)*mult - y_shift_sni,
                                     "Batter", slope_str, slope_req, slope_ok, is_vertical=False, text_pos='top')

            d_sni.set_render_size(800, 550)
            st.image(d_sni.as_svg(), use_container_width=True)

            # Proportions checklist table below
            df_checks = pd.DataFrame([
                {
                    "Parameter Name": "Height (H)",
                    "Parameter Symbol": "H",
                    "Input Value": f"{H_tot:.2f} m",
                    "SNI Requirement": "Reference base",
                    "Status": "INFO"
                },
                {
                    "Parameter Name": "Top Wall Thickness",
                    "Parameter Symbol": "top_wall",
                    "Input Value": f"{top_wall:.2f} m",
                    "SNI Requirement": "≥ 0.30 m",
                    "Status": "PASS" if top_wall_ok else "FAIL"
                },
                {
                    "Parameter Name": "Base Stem Thickness",
                    "Parameter Symbol": "bot_wall",
                    "Input Value": f"{bot_wall:.2f} m",
                    "SNI Requirement": f"≥ 0.1 H ({0.1*H_tot:.2f} m)",
                    "Status": "PASS" if bot_wall_ok else "FAIL"
                },
                {
                    "Parameter Name": "Front Face Batter Slope",
                    "Parameter Symbol": "slope",
                    "Input Value": f"{slope_str} ({slope_val:.4f})",
                    "SNI Requirement": "≥ 1:48 (0.0208)",
                    "Status": "PASS" if slope_ok else "FAIL"
                },
                {
                    "Parameter Name": "Footing Width (B)",
                    "Parameter Symbol": "ftg",
                    "Input Value": f"{ftg:.2f} m",
                    "SNI Requirement": f"0.4 H ~ 0.7 H ({0.4*H_tot:.2f} ~ {0.7*H_tot:.2f} m)",
                    "Status": "PASS" if ftg_ok else "FAIL"
                },
                {
                    "Parameter Name": "Footing Thickness (D)",
                    "Parameter Symbol": "h_ftg",
                    "Input Value": f"{h_ftg:.2f} m",
                    "SNI Requirement": f"H/12 ~ H/10 ({H_tot/12:.2f} ~ {H_tot/10:.2f} m)",
                    "Status": "PASS" if h_ftg_ok else "FAIL"
                },
                {
                    "Parameter Name": "Toe Slab Length",
                    "Parameter Symbol": "toe",
                    "Input Value": f"{toe:.2f} m",
                    "SNI Requirement": f"≥ B/3 ({ftg/3:.2f} m)",
                    "Status": "PASS" if toe_ok else "FAIL"
                }
            ])

            # Style helper
            def style_status(val):
                if val == "PASS":
                    return "color: #27AE60; font-weight: bold;"
                elif val == "FAIL":
                    return "color: #C0392B; font-weight: bold;"
                return "color: #2980B9; font-weight: bold;"

            try:
                styled_df = df_checks.style.map(style_status, subset=["Status"])
            except AttributeError:
                styled_df = df_checks.style.applymap(style_status, subset=["Status"])
            st.table(styled_df)

            # Combined status alert
            all_sni_ok = all([top_wall_ok, bot_wall_ok, slope_ok, ftg_ok, h_ftg_ok, toe_ok])
            if all_sni_ok:
                st.success("🎉 **Success:** All wall geometry parameters meet the typical proportions recommended by **SNI Perencanaan Geoteknik** (Gambar 35b).")
            else:
                st.warning("⚠️ **Warning:** One or more geometry parameters do not satisfy the typical proportions of **SNI Perencanaan Geoteknik** (Gambar 35b). Please adjust the input dimensions to comply.")

        except Exception as ex:
            st.error(f"Error executing SNI Geotechnical verification: {ex}")

    # Card B: Applied Loads & Pressures
    with st.container(border=True):
        st.subheader("🌊 Applied Loads & Pressure Distributions")
        st.markdown('<div class="section-desc">Visual and tabular representation of boundary loading, earth/water pressures, and surcharges.</div>', unsafe_allow_html=True)
        
        # Dropdown list to change type of load displayed
        selected_load_plot = st.selectbox(
            "Select Load Type to Plot",
            [
                "total",
                "soil lateral",
                "soil vertical",
                "hydrostatic",
                "uplift",
                "seismic"
            ],
            index=0,
            key="selected_load_plot"
        )

        # Draw the model dynamically using drawsvg for loads
        try:
            d_load = generate_load_drawing(selected_load_plot)
            d_load.set_render_size(800, 550)
            st.image(d_load.as_svg(), use_container_width=True)
        except Exception as e:
            st.error(f"Error drawing SVG load diagram: {e}")

        # Max Load Metrics Display under the diagram
        st.markdown("### 📊 Peak Loading Parameters")
        
        try:
            soil_lat_force = abs(sum(Fx_soil_q.values()))
        except Exception:
            soil_lat_force = 0.0
            
        try:
            water_lat_force = abs(sum(Fx_water_lat.values()))
        except Exception:
            water_lat_force = 0.0
            
        if selected_load_plot == 'soil lateral':
            h_active = min(Hw, h_soil)
            if surcharge_type == 'Strip Load':
                q_top = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - h_active))['delta sigma x [kPa]'] if h_soil > 0 else 0.0
                q_bot = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
            else:
                q_top = Ka * q if h_soil > 0 else 0.0
                q_bot = Ka * q if h_soil > 0 else 0.0
            p_top = Ka * (gamma_soil_dry * max(0.0, h_soil - h_active)) + q_top if h_soil > 0 else 0.0
            p_bot = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot if h_soil > 0 else 0.0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Active Pressure (Base)", f"{p_bot:.2f} kPa")
                st.metric("Max Passive Resistance (Base)", f"{sigma_p_bot:.2f} kPa", help="Mobilized passive earth pressure resistance (FS = 2.0)")
            with col2:
                top_label = "Active Pressure (Soil Surface)" if h_soil < Hw else "Min Active Pressure (Top)"
                st.metric(top_label, f"{p_top:.2f} kPa")
                st.metric("Passive Resistance (Soil Surface)", f"{sigma_p_top:.2f} kPa", help="Mobilized passive earth pressure resistance (FS = 2.0)")
            with col3:
                net_soil_force = max(0.0, soil_lat_force - P_passive)
                st.metric("Active Lateral Force", f"{soil_lat_force:.2f} kN")
                st.metric("Passive Resisting Force", f"{P_passive:.2f} kN", help="Mobilized passive resistance force (FS = 2.0)")
                st.metric("Net Lateral Soil Force", f"{net_soil_force:.2f} kN", delta=f"-{P_passive:.2f} kN (Resisted)", delta_color="inverse")
                
        elif selected_load_plot == 'soil vertical':
            if surcharge_type == 'Strip Load':
                n_pts = 10
                forces_pts = []
                for k in range(n_pts):
                    frac = (k + 0.5) / n_pts
                    d = frac * heel
                    x_rel = d - offset_surcharge
                    res = stresses_stripload_local(z=h_soil, x=x_rel, width=width_surcharge, imposedstress=q)
                    forces_pts.append(max(0.0, res['delta sigma z [kPa]']))
                avg_q_v = np.mean(forces_pts)
            else:
                avg_q_v = q
            p_v = q_soil + avg_q_v
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Vertical Pressure on Heel", f"{p_v:.2f} kPa")
            with col2:
                st.metric("Total Soil Weight on Heel", f"{abs(soil_total):.2f} kN")
                
        elif selected_load_plot == 'hydrostatic':
            p_w_back = gamma_w * Hwtr
            p_w_front = gamma_w * Hwtr_front
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Water Pressure (Back)", f"{p_w_back:.2f} kPa")
            with col2:
                st.metric("Max Water Pressure (Front)", f"{p_w_front:.2f} kPa" if Hwtr_front > 0 else "0.00 kPa")
            with col3:
                st.metric("Total Hydrostatic Lateral Force", f"{water_lat_force:.2f} kN")
                
        elif selected_load_plot == 'uplift':
            p_uplift = gamma_w * Hwtr
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Uplift Pressure at Base", f"{p_uplift:.2f} kPa")
            with col2:
                st.metric("Total Base Uplift Force", f"{uplift_total:.2f} kN")
                
        elif selected_load_plot == 'seismic':
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Seismic Acceleration (kh)", f"{kh:.4f}")
            with col2:
                st.metric("Total Seismic Inertia Force", f"{F_seismic_pseudo:.2f} kN")
                
        elif selected_load_plot == 'total':
            col1, col2, col3 = st.columns(3)
            with col1:
                if surcharge_type == 'Strip Load':
                    q_bot = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
                else:
                    q_bot = Ka * q
                p_bot = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot
                net_soil_force = max(0.0, soil_lat_force - P_passive)
                st.metric("Max Soil Lateral Pressure", f"{p_bot:.2f} kPa")
                st.metric("Net Soil Lateral Force", f"{net_soil_force:.2f} kN", delta=f"-{P_passive:.2f} kN (Resisted)", delta_color="inverse")
            with col2:
                if surcharge_type == 'Strip Load':
                    n_pts = 10
                    forces_pts = []
                    for k in range(n_pts):
                        frac = (k + 0.5) / n_pts
                        d = frac * heel
                        x_rel = d - offset_surcharge
                        res = stresses_stripload_local(z=h_soil, x=x_rel, width=width_surcharge, imposedstress=q)
                        forces_pts.append(max(0.0, res['delta sigma z [kPa]']))
                    avg_q_v = np.mean(forces_pts)
                else:
                    avg_q_v = q
                p_v = q_soil + avg_q_v
                st.metric("Avg Soil Vertical Pressure", f"{p_v:.2f} kPa")
                st.metric("Soil Vertical Force", f"{abs(soil_total):.2f} kN")
            with col3:
                p_w_back = gamma_w * Hwtr
                st.metric("Max Hydrostatic Pressure", f"{p_w_back:.2f} kPa")
                st.metric("Hydrostatic Lateral Force", f"{water_lat_force:.2f} kN")
            
            st.markdown("---")
            col4, col5, col6 = st.columns(3)
            with col4:
                p_uplift = gamma_w * Hwtr
                st.metric("Uplift Pressure at Base", f"{p_uplift:.2f} kPa")
                st.metric("Total Uplift Force", f"{uplift_total:.2f} kN")
            with col5:
                st.metric("Seismic Accel (kh)", f"{kh:.4f}")
                st.metric("Seismic Inertial Force", f"{F_seismic_pseudo:.2f} kN")
            with col6:
                st.metric("Concrete Self-Weight", f"{W_conc:.2f} kN")
                st.metric("Net Downward Force (heel - uplift)", f"{abs(soil_total) + W_conc - uplift_total:.2f} kN")
            
        # Summary tables in expanders
        with st.expander("📊 Load Summary Tables (Node Forces)"):
            if not df_lateral.empty:
                st.markdown("**Summary Table — Lateral Loads (kN)**")
                st.table(df_lateral)
            if not df_vertical.empty:
                st.markdown("**Summary Table — Vertical Loads (kN)**")
                st.table(df_vertical)
                
        # Detailed Load calculations in expanders
        with st.expander("📝 Show Analytical Load Calculations"):
            # Compute all variables locally to avoid scope issues
            Kp_calc = np.tan(np.radians(45.0 + phi / 2.0)) ** 2
            FS_p = 2.0
            h_active_calc = min(Hw, h_soil)
            
            # Active pressure values
            if surcharge_type == 'Strip Load':
                q_top_calc = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, max(1e-5, h_soil - h_active_calc))['delta sigma x [kPa]'] if h_soil > 0 else 0.0
                q_bot_calc = stresses_stripload_retainingwall_local(q, width_surcharge, offset_surcharge, h_soil, h_soil)['delta sigma x [kPa]']
            else:
                q_top_calc = Ka * q if h_soil > 0 else 0.0
                q_bot_calc = Ka * q if h_soil > 0 else 0.0
            
            pa_top_calc = Ka * (gamma_soil_dry * max(0.0, h_soil - h_active_calc)) + q_top_calc if h_soil > 0 else 0.0
            pa_bot_calc = Ka * (gamma_soil_dry * max(0.0, h_soil - Hwtr) + gamma_soil_wet * min(h_soil, Hwtr)) + q_bot_calc if h_soil > 0 else 0.0
            P_active_calc = 0.5 * (pa_top_calc + pa_bot_calc) * h_active_calc * t if h_soil > 0 else 0.0
            
            # Passive pressure values (mobilized)
            pp_top_calc = (Kp_calc * gamma_soil_dry * h_soil_toe) / FS_p if h_soil_toe > 0 else 0.0
            pp_bot_calc = (Kp_calc * gamma_soil_dry * (h_soil_toe + h_ftg)) / FS_p
            P_passive_calc = 0.5 * (pp_top_calc + pp_bot_calc) * h_ftg * t
            
            # Dry and wet soil heights for active pressure
            h_dry_active = max(0.0, h_soil - Hwtr)
            h_wet_active = min(h_soil, Hwtr)
            
            st.markdown("### 🍂 Earth Pressure Coefficients")
            st.latex(r"K_a = \tan^2\left(45^\circ - \frac{\phi}{2}\right)")
            st.latex(rf"K_a = \tan^2\left(45^\circ - \frac{{{phi:.1f}^\circ}}{{2}}\right) = \tan^2({45.0 - phi/2.0:.1f}^\circ) = {Ka:.4f}")
            st.latex(r"K_p = \tan^2\left(45^\circ + \frac{\phi}{2}\right)")
            st.latex(rf"K_p = \tan^2\left(45^\circ + \frac{{{phi:.1f}^\circ}}{{2}}\right) = \tan^2({45.0 + phi/2.0:.1f}^\circ) = {Kp_calc:.4f}")
            st.info(f"**Ratio $K_p / K_a$ = {Kp_calc/Ka:.1f}** — Passive resistance per unit depth is {Kp_calc/Ka:.0f}× larger than active pressure.")
            
            st.markdown("---")
            st.markdown("### 🔸 Active Earth Pressure (Backfill Side)")
            st.markdown(f"Soil height behind wall: **{h_soil:.2f} m** &nbsp;|&nbsp; Stem height: **{Hw:.2f} m** &nbsp;|&nbsp; Active height: **{h_active_calc:.2f} m**")
            st.markdown(f"Dry soil height: **{h_dry_active:.2f} m** &nbsp;|&nbsp; Wet soil height: **{h_wet_active:.2f} m**")
            
            st.markdown("**At top of active zone:**")
            st.latex(rf"p_{{active,top}} = K_a \cdot \gamma_{{dry}} \cdot 0 + K_a \cdot q = {Ka:.4f} \times {q:.1f} = {q_top_calc:.2f}\;\text{{kPa}}")
            
            st.markdown("**At base of stem:**")
            st.latex(rf"p_{{active,bot}} = K_a \left(\gamma_{{dry}} \cdot h_{{dry}} + \gamma_{{wet}} \cdot h_{{wet}}\right) + K_a \cdot q")
            sigma_soil_dry_part = gamma_soil_dry * h_dry_active
            sigma_soil_wet_part = gamma_soil_wet * h_wet_active
            st.latex(rf"= {Ka:.4f} \times \left({gamma_soil_dry:.1f} \times {h_dry_active:.2f} + {gamma_soil_wet:.1f} \times {h_wet_active:.2f}\right) + {q_bot_calc:.2f}")
            st.latex(rf"= {Ka:.4f} \times \left({sigma_soil_dry_part:.2f} + {sigma_soil_wet_part:.2f}\right) + {q_bot_calc:.2f}")
            st.latex(rf"= {Ka:.4f} \times {sigma_soil_dry_part + sigma_soil_wet_part:.2f} + {q_bot_calc:.2f} = {pa_bot_calc:.2f}\;\text{{kPa}}")
            
            st.markdown("**Total Active Force:**")
            st.latex(rf"P_{{active}} = \frac{{1}}{{2}} (p_{{top}} + p_{{bot}}) \times h_{{active}} \times t")
            st.latex(rf"= \frac{{1}}{{2}} \times ({pa_top_calc:.2f} + {pa_bot_calc:.2f}) \times {h_active_calc:.2f} \times {t:.2f}")
            st.latex(rf"= \boxed{{{P_active_calc:.2f}\;\text{{kN}}}}")
            
            st.markdown("---")
            st.markdown("### 🟢 Mobilized Passive Earth Pressure (Toe Side)")
            st.markdown(f"Soil above toe: **{h_soil_toe:.2f} m** &nbsp;|&nbsp; Footing thickness: **{h_ftg:.2f} m** &nbsp;|&nbsp; Mobilization FS: **{FS_p:.1f}**")
            st.markdown("Passive pressure only acts on the **footing front face** (height = footing thickness).")
            
            st.markdown("**At footing top (depth = h_soil_toe):**")
            pp_top_unreduced = Kp_calc * gamma_soil_dry * h_soil_toe
            st.latex(rf"p_{{passive,top}} = \frac{{K_p \cdot \gamma_{{dry}} \cdot h_{{soil,toe}}}}{{FS_{{passive}}}}")
            st.latex(rf"= \frac{{{Kp_calc:.4f} \times {gamma_soil_dry:.1f} \times {h_soil_toe:.2f}}}{{{FS_p:.1f}}} = \frac{{{pp_top_unreduced:.2f}}}{{{FS_p:.1f}}} = {pp_top_calc:.2f}\;\text{{kPa}}")
            
            st.markdown("**At footing bottom (depth = h_soil_toe + h_ftg):**")
            pp_bot_unreduced = Kp_calc * gamma_soil_dry * (h_soil_toe + h_ftg)
            st.latex(rf"p_{{passive,bot}} = \frac{{K_p \cdot \gamma_{{dry}} \cdot (h_{{soil,toe}} + h_{{ftg}})}}{{FS_{{passive}}}}")
            st.latex(rf"= \frac{{{Kp_calc:.4f} \times {gamma_soil_dry:.1f} \times ({h_soil_toe:.2f} + {h_ftg:.2f})}}{{{FS_p:.1f}}} = \frac{{{pp_bot_unreduced:.2f}}}{{{FS_p:.1f}}} = {pp_bot_calc:.2f}\;\text{{kPa}}")
            
            st.markdown("**Total Passive Force (on footing face only):**")
            st.latex(rf"P_{{passive}} = \frac{{1}}{{2}} (p_{{top}} + p_{{bot}}) \times h_{{ftg}} \times t")
            st.latex(rf"= \frac{{1}}{{2}} \times ({pp_top_calc:.2f} + {pp_bot_calc:.2f}) \times {h_ftg:.2f} \times {t:.2f}")
            st.latex(rf"= \boxed{{{P_passive_calc:.2f}\;\text{{kN}}}}")
            
            st.markdown("---")
            st.markdown("### 🌊 Hydrostatic Water Lateral Force")
            st.latex(rf"P_{{water}} = \frac{{1}}{{2}} \gamma_w H_{{wtr}}^2 \cdot t = \frac{{1}}{{2}} \times {gamma_w:.2f} \times {Hwtr:.2f}^2 \times {t:.2f} = {water_lat_force:.2f}\;\text{{kN}}")
            
            st.markdown("---")
            st.markdown("### ⚖️ Net Lateral Force Summary")
            P_net_calc = soil_lat_force + water_lat_force - P_passive_calc
            st.latex(r"P_{net} = P_{active} + P_{water} - P_{passive}")
            st.latex(rf"= {soil_lat_force:.2f} + {water_lat_force:.2f} - {P_passive_calc:.2f}")
            st.latex(rf"= \boxed{{{P_net_calc:.2f}\;\text{{kN}}}}")
            
            st.markdown(f"- **Total soil vertical load (on heel):** `{soil_total:.3f} kN`")
            
            st.markdown("---")
            st.markdown("#### Concrete Self-Weight")
            st.latex(r"W_{conc} = \gamma_c\,A_{conc}\,t")
            st.markdown(f"- **Concrete Weight (W_conc):** {W_conc:.3f} kN")
            st.markdown(f"- **Stem Area:** {wall_area:.3f} m², &nbsp; **Footing Area:** {footing_area:.3f} m²")
            
            st.markdown("---")
            st.markdown("#### Water Uplift Pressure")
            st.latex(r"F_{uplift,node} = (\gamma_w H_{wtr})\cdot L\cdot t")
            st.markdown(f"- **Water uplift force:** {uplift_total:.3f} kN")
            st.markdown(f"- **Net vertical load:** {net_vertical:.3f} kN")
            
            st.markdown("---")
            st.markdown("#### Seismic Loading (Pseudo-static)")
            st.latex(r"F_{seismic} = k_h\,(W_{soil} + W_{conc})")
            st.markdown(f"- **Seismic Coefficient ($k_h$):** {kh:.4f}")
            st.markdown(f"- **Pseudo-static lateral force:** {F_seismic_pseudo:.3f} kN")
            st.markdown(f"- **Transient Peak Acceleration estimate:** {peak_accel:.3f} m/s²")
            st.markdown(f"- **Inertial Peak Force estimate:** {inertial_peak:.3f} kN")


# --- RIGHT COLUMN: ANALYSIS RESULTS ---
with col_right:

    # Remove zero-length support elements and anchor nodes to prevent errors in opsvis plotting
    for idx in range(len(bottom_node_ids)):
        elem_tag = spring_elem_base_id + idx
        anchor_nid = anchor_node_base_id + idx
        try:
            ops.remove('element', elem_tag)
            ops.remove('node', anchor_nid)
        except Exception:
            pass

    # Remove shear key spring elements and anchor nodes for clean opsvis plotting
    if include_shear_key:
        for idx_sk in range(len(shear_key_bottom_nids)):
            try:
                ops.remove('element', sk_spring_elem_base + idx_sk)
                ops.remove('node', sk_anchor_base + idx_sk)
            except Exception:
                pass

    # Card C: Stem Bending Moment Diagram
    with st.container(border=True):
        st.subheader("📉 Stem Bending Moment Diagram")
        st.markdown('<div class="section-desc">Internal bending moment contour along the height of the stem wall.</div>', unsafe_allow_html=True)
        
        try:
            node_moments_dict = get_stem_moments()
            # Use actual node tags from current model state (after anchor removal)
            active_node_tags = ops.getNodeTags()
            nds_val_moment = np.zeros(len(active_node_tags))
            for idx_nd, ntag in enumerate(active_node_tags):
                if ntag in node_moments_dict:
                    nds_val_moment[idx_nd] = node_moments_dict[ntag]
                else:
                    nds_val_moment[idx_nd] = 0.0
            
            plt.close('all')
            opsv.plot_stress_2d(nds_val_moment)
            fig_m = plt.gcf()
            fig_m.set_size_inches(6, 4.5)
            try:
                ax = fig_m.axes[0]
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_title('Stem Node Moment (kN·m)')
                ax.relim()
                ax.autoscale_view()
            except Exception:
                pass
            import os
            os.makedirs("report_temp", exist_ok=True)
            fig_m.tight_layout()
            fig_m.savefig("report_temp/stem_bending_moment.png", dpi=150, bbox_inches='tight')
            st.pyplot(fig_m)
            plt.close(fig_m)
        except Exception as e:
            st.error(f'Failed to compute/plot stem node moments: {e}')

    # Card D: Finite Element Stress Contours
    with st.container(border=True):
        st.subheader("🎭 Finite Element Stress Contours")
        st.markdown('<div class="section-desc">2D stress contour plots mapped across the retaining wall mesh elements.</div>', unsafe_allow_html=True)
        
        stress_col1, stress_col2 = st.columns(2)
        
        with stress_col1:
            st.markdown("**X-Direction stress ($\sigma_{xx}$)**")
            plt.close('all')
            try:
                sig_out = opsv.sig_out_per_node()
                j, jstr = 0, 'σ_xx (kN/m^2)'
                nds_val = np.array(sig_out)[:, j]

                opsv.plot_stress_2d(nds_val)
                fig_res = plt.gcf()
                fig_res.set_size_inches(5.6, 4.2)
                try:
                    ax = fig_res.axes[0]
                    ax.set_xlabel('x [m]')
                    ax.set_ylabel('y [m]')
                    ax.relim()
                    ax.autoscale_view()
                except Exception:
                    pass
                fig_res.tight_layout()
                fig_res.savefig("report_temp/stress_xx.png", dpi=150, bbox_inches='tight')
                st.pyplot(fig_res)
                plt.close(fig_res)
            except Exception as e:
                st.error(f'Failed to render σ_xx: {e}')
                
        with stress_col2:
            st.markdown("**Y-Direction stress ($\sigma_{yy}$)**")
            plt.close('all')
            try:
                sig_out = opsv.sig_out_per_node()
                j, jstr = 1, 'σ_yy (kN/m^2)'
                nds_val = np.array(sig_out)[:, j]

                opsv.plot_stress_2d(nds_val)
                fig_res2 = plt.gcf()
                fig_res2.set_size_inches(5.6, 4.2)
                try:
                    ax2 = fig_res2.axes[0]
                    ax2.set_xlabel('x [m]')
                    ax2.set_ylabel('y [m]')
                    ax2.relim()
                    ax2.autoscale_view()
                except Exception:
                    pass
                fig_res2.tight_layout()
                fig_res2.savefig("report_temp/stress_yy.png", dpi=150, bbox_inches='tight')
                st.pyplot(fig_res2)
                plt.close(fig_res2)
            except Exception as e:
                st.error(f'Failed to render σ_yy: {e}')

    # Card E: Detailed Stability Verification
    with st.container(border=True):
        st.subheader("🛡️ Stability Verification Report")
        st.markdown('<div class="section-desc">Detailed overturning, sliding, and stem junction stability verification checklist.</div>', unsafe_allow_html=True)
        
        ot_limit = 1.5
        slide_limit = 1.5
        
        def status_badge_html(val, limit):
            if val is None:
                return "<span style='color:gray; font-weight:bold;'>N/A</span>"
            if val == float('inf'):
                return "<span style='background-color:#28a745; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px; text-transform:uppercase;'>PASS (∞)</span>"
            if val >= limit:
                return f"<span style='background-color:#28a745; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px; text-transform:uppercase;'>PASS ({val:.3f})</span>"
            else:
                return f"<span style='background-color:#dc3545; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px; text-transform:uppercase;'>FAIL ({val:.3f})</span>"

        fs_ot_global_val = f"{FS_ot_global:.3f}" if FS_ot_global is not None else "0.000"
        fs_slide_global_val = f"{FS_slide_global:.3f}" if FS_slide_global is not None else "0.000"
        fs_ot_stem_val = f"{FS_ot_stem:.3f}" if FS_ot_stem is not None else "0.000"

        report_md = f"""
        - **Global Overturning Safety** (pivot $x=0.0$):
          - $M_{{driving}}$ = **{M_ot_global:.3f} kN·m** &nbsp; | &nbsp; $M_{{resisting}}$ = **{M_res_global:.3f} kN·m**
          - **FS = {fs_ot_global_val}** &nbsp;&nbsp; {status_badge_html(FS_ot_global, ot_limit)}
        
        - **Global Sliding Safety**:
          - $R_{{sliding}}$ = **{R_slide:.3f} kN** &nbsp; | &nbsp; $F_{{driving, net}}$ = **{F_drive:.3f} kN** (reduced by $P_{{passive}}$ = **{P_passive:.3f} kN**)
          - **FS = {fs_slide_global_val}** &nbsp;&nbsp; {status_badge_html(FS_slide_global, slide_limit)}
          
        - **Stem Junction Overturning Safety** (pivot $x={x_pivot_stem:.3f}$):
          - $M_{{driving}}$ = **{M_ot_stem:.3f} kN·m** &nbsp; | &nbsp; $M_{{resisting}}$ = **{M_res_stem:.3f} kN·m**
          - **FS = {fs_ot_stem_val}** &nbsp;&nbsp; {status_badge_html(FS_ot_stem, ot_limit)}
        """
        if enable_bearing:
            fs_bearing_val = f"{FS_actual_bearing:.3f}" if FS_actual_bearing != float('inf') else "∞"
            report_md += f"""
        - **Bearing Capacity Safety**:
          - $\\sigma_{{max}}$ = **{sigma_max:.3f} kPa** &nbsp; | &nbsp; $q_{{allow}}$ = **{q_allow:.3f} kPa** (SF = {FS_bearing:.2f})
          - **FS = {fs_bearing_val}** &nbsp;&nbsp; {status_badge_html(FS_actual_bearing, FS_bearing)}
        
        - **Consolidation Settlement (SNI 8460:2017)**:
          - $s_{{consol}}$ = **{consol_settlement_mm:.2f} mm** &nbsp; | &nbsp; $s_{{allow}}$ = **{consol_allow_mm:.1f} mm** (15cm + B/600)
          - **Status:** {"<span style='background-color:#28a745; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px; text-transform:uppercase;'>PASS</span>" if consol_pass else "<span style='background-color:#dc3545; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px; text-transform:uppercase;'>FAIL</span>"}
        """
        st.markdown(report_md, unsafe_allow_html=True)
        
        with st.expander("📖 View Overturning and Sliding Equations"):
            st.markdown("**Global Overturning Verification**")
            st.latex(r"M_{ot} = \sum |F_{x,i} \cdot z_i|")
            st.latex(r"M_{res} = W_{base}\cdot x_{base} + W_{stem,rect}\cdot x_{stem,rect} + W_{stem,tri}\cdot x_{stem,tri} + W_{soil}\cdot x_{soil} - M_{uplift}")
            st.latex(r"FS_{ot} = \frac{M_{res}}{M_{ot}} \ge 1.5")
            
            st.markdown("**Global Sliding Verification**")
            st.latex(r"R_{slide} = W_{down} \cdot \tan(\phi)")
            st.latex(r"F_{drive, net} = F_{active} - P_{passive}")
            st.latex(r"FS_{slide} = \frac{R_{slide}}{F_{drive, net}} \ge 1.5")
            
            st.markdown("**Stem Overturning Verification**")
            st.latex(r"M_{ot} = \sum |F_{x,i} \cdot (x_i - x_{pivot})|")
            st.latex(r"M_{res} = W_{down} \cdot |x_{centroid} - x_{pivot}|")
            st.latex(r"FS_{ot} = \frac{M_{res}}{M_{ot}} \ge 1.5")

            if enable_bearing:
                st.markdown("**Bearing Capacity & Consolidation Settlement**")
                st.latex(r"\sigma_{max} = \max(q_{toe}, q_{heel})")
                st.latex(r"q_{allow} = \frac{q_{ult}}{FS_{bearing}}")
                st.latex(r"FS_{actual} = \frac{q_{ult}}{\sigma_{max}} \ge FS_{bearing}")
                st.markdown("**Consolidation Settlement Formulation:**")
                st.latex(r"s_{consol} = \frac{H_0}{1 + e_0} C_c \log_{10} \frac{\sigma'_{v0} + \Delta\sigma'_v}{\sigma'_{v0}}\quad\text{(Normally Consolidated)}")
                st.latex(r"s_{allow} = 15\text{ cm} + \frac{B\text{ (cm)}}{600}\quad\text{(Allowable Limit per SNI 8460:2017)}")

    # Card E.2: FEM Settlement & Displacement Report (Winkler Springs)
    with st.container(border=True):
        st.subheader("📏 FEM Settlement & Displacement Report")
        st.markdown('<div class="section-desc">Vertical and horizontal displacements at the base of the footing slab. Allowable limits from SNI 8460:2017.</div>', unsafe_allow_html=True)
        
        status_diff_html = "<span style='background-color:#28a745; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px; text-transform:uppercase;'>PASS</span>" if diff_settlement_pass else "<span style='background-color:#dc3545; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px; text-transform:uppercase;'>FAIL</span>"
        
        st.markdown(f"""
        - **Toe Settlement (Node {bottom_node_ids[0]}):** **{settlement_toe_mm:.2f} mm** (X-disp: {disp_toe_x*1000:.2f} mm, Y-disp: {disp_toe_y*1000:.2f} mm)
        - **Heel Settlement (Node {bottom_node_ids[-1]}):** **{settlement_heel_mm:.2f} mm** (X-disp: {disp_heel_x*1000:.2f} mm, Y-disp: {disp_heel_y*1000:.2f} mm)
        - **Differential Settlement (Δs):** **{diff_settlement_mm:.2f} mm** (Allowable: **{allowable_diff_settlement_mm:.1f} mm**) {status_diff_html}
        - **Footing Slab Rotation:** **{rotation_deg:.4f}°** ({rotation_rad:.6f} rad)
        """, unsafe_allow_html=True)
        
        if not df_bottom_disp.empty:
            with st.expander("📊 View Detailed Bottom Node Displacements (X and Y)"):
                st.dataframe(df_bottom_disp, use_container_width=True)

    # Card F: Concrete Strength Verification
    with st.container(border=True):
        st.subheader("🛡️ Concrete Strength Verification")
        st.markdown('<div class="section-desc">Comparison of structural bending moment demands against ultimate cross-section capacities.</div>', unsafe_allow_html=True)
        
        def check_badge(is_pass):
            if is_pass:
                return "<span style='background-color:#28a745; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px;'>PASS</span>"
            else:
                return "<span style='background-color:#dc3545; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:12px;'>FAIL</span>"
                
        # Comparison Table using standard HTML
        st.markdown(f"""
        <table style="width:100%; border-collapse: collapse; margin-bottom: 15px;">
            <thead>
                <tr style="border-bottom: 2px solid #eef1f6; text-align: left;">
                    <th style="padding: 8px;">Component</th>
                    <th style="padding: 8px;">Bending Type</th>
                    <th style="padding: 8px; text-align: right;">Actual M* (kN·m)</th>
                    <th style="padding: 8px; text-align: right;">Capacity φMu (kN·m)</th>
                    <th style="padding: 8px; text-align: center;">Status</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #eef1f6;">
                    <td style="padding: 8px; font-weight: 600;">Stem Wall (Base)</td>
                    <td style="padding: 8px; color: #6c757d; font-size: 0.9rem;">Bending major</td>
                    <td style="padding: 8px; text-align: right; font-family: monospace;">{M_actual_stem:.2f}</td>
                    <td style="padding: 8px; text-align: right; font-family: monospace;">{M_u_stem_allow:.2f}</td>
                    <td style="padding: 8px; text-align: center;">{check_badge(stem_pass)}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef1f6;">
                    <td style="padding: 8px; font-weight: 600;">Footing Toe (Slab)</td>
                    <td style="padding: 8px; color: #6c757d; font-size: 0.9rem;">Sagging (Tension Bottom)</td>
                    <td style="padding: 8px; text-align: right; font-family: monospace;">{M_actual_toe:.2f}</td>
                    <td style="padding: 8px; text-align: right; font-family: monospace;">{M_u_ftg_sag_allow:.2f}</td>
                    <td style="padding: 8px; text-align: center;">{check_badge(toe_pass)}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef1f6;">
                    <td style="padding: 8px; font-weight: 600;">Footing Heel (Slab)</td>
                    <td style="padding: 8px; color: #6c757d; font-size: 0.9rem;">Hogging (Tension Top)</td>
                    <td style="padding: 8px; text-align: right; font-family: monospace;">{M_actual_heel:.2f}</td>
                    <td style="padding: 8px; text-align: right; font-family: monospace;">{M_u_ftg_hog_allow:.2f}</td>
                    <td style="padding: 8px; text-align: center;">{check_badge(heel_pass)}</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
        # Conclusion box
        all_concrete_pass = stem_pass and toe_pass and heel_pass
        conclusion_info = f" (using Main Rebar **D{stem_rebar_dia}** for Stem Wall, and **D{ftg_rebar_dia}** for Footing Slabs)"
        if all_concrete_pass:
            st.success(f"🤖 **Conclusion**: The structural verification is successful! All key components (Stem Wall, Footing Toe, and Footing Heel) satisfy the bending capacity checks. The current reinforcement configuration{conclusion_info} is **adequate** and **safe** under the design limit state load combinations.")
        else:
            failed_components = []
            if not stem_pass: failed_components.append("Stem Wall")
            if not toe_pass: failed_components.append("Footing Toe")
            if not heel_pass: failed_components.append("Footing Heel")
            st.error(f"⚠️ **Conclusion**: Structural verification **FAILED** due to insufficient bending capacity in: **{', '.join(failed_components)}**. The current reinforcement configuration{conclusion_info} is **inadequate**. Please consider increasing the rebar diameter, decreasing rebar spacing, or increasing the thickness of the failing member.")

        # Expander to see layouts
        with st.expander("🔍 View Concrete Rebar Cross Sections"):
            if fig_stem_sec is not None:
                st.pyplot(fig_stem_sec)
            if fig_ftg_sec is not None:
                st.pyplot(fig_ftg_sec)

# ----------------------------------------------------
# Pile Foundation Analysis Results Section
# ----------------------------------------------------
if enable_pile:
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #1f4068;'>🧱 Pile Foundation Analysis & Design</h2>", unsafe_allow_html=True)
    st.markdown('<div class="section-desc" style="text-align: center; margin-bottom: 24px;">Comprehensive evaluation of base anchor piles under combined vertical load, lateral shear, and bending demand.</div>', unsafe_allow_html=True)
    
    pile_kpi_cols = st.columns(6)
    with pile_kpi_cols[0]:
        with st.container(border=True):
            st.metric("Toe Pile Axial Force", f"{P_toe:.2f} kN", delta="Compression" if P_toe >= 0 else "Tension", delta_color="normal" if P_toe >= 0 else "inverse")
    with pile_kpi_cols[1]:
        with st.container(border=True):
            st.metric("Toe Pile Max Shear", f"{V_max_toe:.2f} kN")
    with pile_kpi_cols[2]:
        with st.container(border=True):
            st.metric("Toe Pile Max Moment", f"{M_max_toe:.2f} kN·m")
    with pile_kpi_cols[3]:
        with st.container(border=True):
            st.metric("Heel Pile Axial Force", f"{P_heel:.2f} kN", delta="Compression" if P_heel >= 0 else "Tension", delta_color="normal" if P_heel >= 0 else "inverse")
    with pile_kpi_cols[4]:
        with st.container(border=True):
            st.metric("Heel Pile Max Shear", f"{V_max_heel:.2f} kN")
    with pile_kpi_cols[5]:
        with st.container(border=True):
            st.metric("Heel Pile Max Moment", f"{M_max_heel:.2f} kN·m")

    # Layout for plots
    plot_col1, plot_col2, plot_col3 = st.columns([1.2, 1, 1])
    
    with plot_col1:
        with st.container(border=True):
            st.subheader("📐 Pile Foundation Drawing")
            st.markdown('<div class="section-desc">Retaining wall section showing embedded base anchors with reaction forces.</div>', unsafe_allow_html=True)
            # Create SVG drawing of piles below footing base
            try:
                mult = 1000
                max_h = max(Hw, h_soil, h_soil_toe, Hwtr, Hwtr_front) + h_ftg
                # Set display parameters to shrink canvas borders and margins
                L_draw = 1.2
                margin_bottom = 0.4
                margin_top = 0.4
                margin_left = 0.4
                margin_right = 0.4
                
                W_canvas = ftg + margin_left + margin_right
                H_canvas = max_h + L_draw + margin_bottom + margin_top
                
                d_pile_svg = draw.Drawing(W_canvas * mult, H_canvas * mult, origin='bottom-left')
                
                x_shift = margin_left * mult
                yshift_pile = (margin_bottom + L_draw) * mult
                
                # Draw soil - Dry soil (above water table)
                if 'soil_dry_poly' in globals() and soil_dry_poly:
                    pts_dry = []
                    for px, py in soil_dry_poly:
                        pts_dry.extend([px*mult + x_shift, py*mult - yshift_pile])
                    d_pile_svg.append(draw.Lines(*pts_dry, close=True, fill='#5BC2A5', stroke='black', stroke_width=3))

                if 'soil_wet_poly' in globals() and soil_wet_poly:
                    pts_wet = []
                    for px, py in soil_wet_poly:
                        pts_wet.extend([px*mult + x_shift, py*mult - yshift_pile])
                    d_pile_svg.append(draw.Lines(*pts_wet, close=True, fill='#A6F527', stroke='black', stroke_width=3))

                if h_soil_toe > 0:
                    d_pile_svg.append(draw.Lines(x_shift + 0, -h_ftg*mult - yshift_pile,
                                            x_shift + toe*mult, -h_ftg*mult - yshift_pile,
                                            x_shift + toe*mult, -(h_ftg + h_soil_toe)*mult - yshift_pile,
                                            x_shift + 0, -(h_ftg + h_soil_toe)*mult - yshift_pile,
                                            close=True, fill='#5BC2A5', stroke='black', stroke_width=3))

                # Draw Retaining wall (shifted up by L_draw + margin_bottom)
                wall_points_pile = [
                    x_shift + 0.0,  -yshift_pile,
                    x_shift + 0.0,  -h_ftg*mult - yshift_pile,
                    x_shift + toe*mult, -h_ftg*mult - yshift_pile
                ]
                for px, py in stem_pts:
                    wall_points_pile.extend([x_shift + px*mult, py*mult - yshift_pile])
                wall_points_pile.extend([
                    x_shift + ftg*mult, -h_ftg*mult - yshift_pile,
                    x_shift + ftg*mult, -yshift_pile
                ])
                d_pile_svg.append(draw.Lines(*wall_points_pile, close=True, fill='#eeee00', stroke='black', stroke_width=50))
                
                # Draw Piles with zig-zag break lines
                w_draw = (diameter_pile if pile_shape == 'Circle' else width_x_pile)
                y_top = -yshift_pile
                y_bot = -margin_bottom * mult
                zag1_y = y_bot + 0.05 * mult
                zag2_y = y_bot - 0.05 * mult
                
                # Toe Pile
                x_center_toe = x_shift + pile_offset * mult
                x_l_toe = x_center_toe - (w_draw / 2.0) * mult
                x_r_toe = x_center_toe + (w_draw / 2.0) * mult
                d_pile_svg.append(draw.Lines(
                    x_l_toe, y_top,
                    x_r_toe, y_top,
                    x_r_toe, y_bot,
                    x_l_toe + (w_draw * 0.75) * mult, zag1_y,
                    x_l_toe + (w_draw * 0.25) * mult, zag2_y,
                    x_l_toe, y_bot,
                    close=True,
                    fill='#CCCCCC' if pile_material == 'Concrete' else '#777777', 
                    stroke='black', stroke_width=20
                ))
                
                # Heel Pile
                x_center_heel = x_shift + (ftg - pile_offset) * mult
                x_l_heel = x_center_heel - (w_draw / 2.0) * mult
                x_r_heel = x_center_heel + (w_draw / 2.0) * mult
                d_pile_svg.append(draw.Lines(
                    x_l_heel, y_top,
                    x_r_heel, y_top,
                    x_r_heel, y_bot,
                    x_l_heel + (w_draw * 0.75) * mult, zag1_y,
                    x_l_heel + (w_draw * 0.25) * mult, zag2_y,
                    x_l_heel, y_bot,
                    close=True,
                    fill='#CCCCCC' if pile_material == 'Concrete' else '#777777', 
                    stroke='black', stroke_width=20
                ))
                
                # Draw pile reaction force arrows at the top of the piles
                def draw_reaction_arrow(d_draw, x_c, p_val):
                    color = 'red' if p_val >= 0 else 'blue'
                    y_top = -yshift_pile
                    if p_val >= 0:
                        d_draw.append(draw.Line(x_c, y_top + 300, x_c, y_top + 50, stroke=color, stroke_width=15))
                        d_draw.append(draw.Lines(x_c - 50, y_top + 150, x_c, y_top + 50, x_c + 50, y_top + 150, stroke=color, stroke_width=15))
                    else:
                        d_draw.append(draw.Line(x_c, y_top + 50, x_c, y_top + 300, stroke=color, stroke_width=15))
                        d_draw.append(draw.Lines(x_c - 50, y_top + 200, x_c, y_top + 300, x_c + 50, y_top + 200, stroke=color, stroke_width=15))
                    d_draw.append(draw.Text(f"{abs(p_val):.1f} kN", 180, x_c + 120, y_top + 220, fill='black', font_weight='bold'))
                
                draw_reaction_arrow(d_pile_svg, x_center_toe, P_toe)
                draw_reaction_arrow(d_pile_svg, x_center_heel, P_heel)
                
                # Labels
                d_pile_svg.append(draw.Text("TOE", 180, x_center_toe, y_top + 600, fill='black', text_anchor='middle', font_weight='bold'))
                d_pile_svg.append(draw.Text("HEEL", 180, x_center_heel, y_top + 600, fill='black', text_anchor='middle', font_weight='bold'))
                
                d_pile_svg.set_render_size(600, 500)
                st.image(d_pile_svg.as_svg(), use_container_width=True)
            except Exception as e_svg:
                st.error(f"Error drawing piles SVG: {e_svg}")
                
    with plot_col2:
        with st.container(border=True):
            st.subheader("📈 Winkler Response Profiles")
            st.markdown('<div class="section-desc">Pile deflection, shear force, bending moment, and axial settlement profiles.</div>', unsafe_allow_html=True)
            try:
                fig_winkler, (ax_d, ax_v, ax_m, ax_s) = plt.subplots(1, 4, figsize=(12, 8), sharey=True)
                
                elevs_defl_toe = res_toe.deflection['Elevation [m]']
                defl_toe = res_toe.deflection['Deflection [m]'] * 1000.0
                elevs_forces_toe = res_toe.forces['Elevation [m]']
                shear_toe = res_toe.forces['V [kN]']
                moment_toe = res_toe.forces['M [kNm]']
                
                elevs_defl_heel = res_heel.deflection['Elevation [m]']
                defl_heel = res_heel.deflection['Deflection [m]'] * 1000.0
                elevs_forces_heel = res_heel.forces['Elevation [m]']
                shear_heel = res_heel.forces['V [kN]']
                moment_heel = res_heel.forces['M [kNm]']
                
                # Axial settlement profiles
                elevs_settle_toe = res_toe.settlement['Elevation [m]']
                settle_toe = abs(res_toe.settlement['Settlement [m]']) * 1000.0
                elevs_settle_heel = res_heel.settlement['Elevation [m]']
                settle_heel = abs(res_heel.settlement['Settlement [m]']) * 1000.0
                
                ax_d.plot(defl_toe, elevs_defl_toe, 'b-', label='Toe Pile')
                ax_d.plot(defl_heel, elevs_defl_heel, 'r--', label='Heel Pile')
                ax_d.set_xlabel('Deflection (mm)')
                ax_d.set_ylabel('Elevation (m)')
                ax_d.grid(True)
                ax_d.legend(loc='lower right')
                
                ax_v.plot(shear_toe, elevs_forces_toe, 'b-', label='Toe Pile')
                ax_v.plot(shear_heel, elevs_forces_heel, 'r--', label='Heel Pile')
                ax_v.set_xlabel('Shear Force (kN)')
                ax_v.grid(True)
                
                ax_m.plot(moment_toe, elevs_forces_toe, 'b-', label='Toe Pile')
                ax_m.plot(moment_heel, elevs_forces_heel, 'r--', label='Heel Pile')
                ax_m.set_xlabel('Moment (kN·m)')
                ax_m.grid(True)
                
                ax_s.plot(settle_toe, elevs_settle_toe, 'b-', label='Toe Pile')
                ax_s.plot(settle_heel, elevs_settle_heel, 'r--', label='Heel Pile')
                ax_s.set_xlabel('Axial Settlement (mm)')
                ax_s.grid(True)
                
                fig_winkler.tight_layout()
                try:
                    import os
                    os.makedirs("report_temp", exist_ok=True)
                    fig_winkler.savefig("report_temp/pile_winkler.png", dpi=150, bbox_inches='tight')
                except Exception:
                    pass
                st.pyplot(fig_winkler)
            except Exception as e_w:
                st.error(f"Error plotting Winkler response: {e_w}")
                
    with plot_col3:
        with st.container(border=True):
            st.subheader("🎯 Moment-Axial Interaction")
            st.markdown('<div class="section-desc">Pile design capacity envelope showing the demand point for each pile.</div>', unsafe_allow_html=True)
            try:
                fig_int, ax_int = plt.subplots(figsize=(6, 5))
                ax_int.plot(curve_m, curve_n, 'k-', linewidth=2, label='Capacity Envelope')
                ax_int.scatter([M_max_toe], [P_toe], color='blue', marker='o', s=100, label=f'Toe Pile (M={M_max_toe:.1f}, P={P_toe:.1f})')
                ax_int.scatter([M_max_heel], [P_heel], color='red', marker='^', s=100, label=f'Heel Pile (M={M_max_heel:.1f}, P={P_heel:.1f})')
                ax_int.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax_int.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                ax_int.set_xlabel('Bending Moment (kN·m)')
                ax_int.set_ylabel('Axial Force (kN)')
                ax_int.set_title('Pile Bending-Axial Interaction')
                ax_int.legend(loc='upper right')
                ax_int.grid(True)
                fig_int.tight_layout()
                try:
                    import os
                    os.makedirs("report_temp", exist_ok=True)
                    fig_int.savefig("report_temp/pile_interaction.png", dpi=150, bbox_inches='tight')
                except Exception:
                    pass
                st.pyplot(fig_int)
                plt.close(fig_int)
            except Exception as e_i:
                st.error(f"Error plotting Interaction Diagram: {e_i}")

    # Pile Cross Section Geometry
    st.markdown("---")
    st.subheader("🔍 Pile Cross Section Geometry")
    if fig_pile_sec is not None:
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.pyplot(fig_pile_sec)
        with col_p2:
            if pile_material == 'Concrete':
                if pile_shape == 'Circle':
                    st.markdown(f"""
                    **Circular Concrete Pile Section**:
                    - **Pile Diameter**: `{diameter_pile:.3f} m` ({diameter_pile*1000:.0f} mm)
                    - **Concrete Cover**: `{cover_pile:.1f} mm`
                    - **Reinforcement**: **{int(n_rebar_pile)}** main bars of **D{rebar_dia_pile}**
                    - **Friction Array Radius**: `{diameter_pile*1000/2 - cover_pile:.1f} mm`
                    """)
                else:
                    st.markdown(f"""
                    **Rectangular Concrete Pile Section**:
                    - **Pile Dimensions (dx × dy)**: `{width_x_pile:.3f} m × {width_y_pile:.3f} m` ({width_x_pile*1000:.0f} mm × {width_y_pile*1000:.0f} mm)
                    - **Concrete Cover**: `{cover_pile:.1f} mm`
                    - **Reinforcement**: **4** corner bars of **D{rebar_dia_pile}**
                    """)
            else:
                if pile_shape == 'Circle':
                    st.markdown(f"""
                    **Solid Steel Circular Pile Section**:
                    - **Pile Diameter**: `{diameter_pile:.3f} m` ({diameter_pile*1000:.0f} mm)
                    """)
                else:
                    st.markdown(f"""
                    **Solid Steel Rectangular Pile Section**:
                    - **Pile Dimensions (dx × dy)**: `{width_x_pile:.3f} m × {width_y_pile:.3f} m` ({width_x_pile*1000:.0f} mm × {width_y_pile*1000:.0f} mm)
                    """)

    # Pile Foundation Report Cards
    st.subheader("📋 Pile Foundation Verification Report")
    
    st.markdown(f"""
    <table style="width:100%; border-collapse: collapse; margin-bottom: 25px;">
        <thead>
            <tr style="border-bottom: 2px solid #eef1f6; text-align: left;">
                <th style="padding: 10px;">Check Item</th>
                <th style="padding: 10px;">Demand (Actual)</th>
                <th style="padding: 10px;">Capacity (Allowable)</th>
                <th style="padding: 10px; text-align: center;">Safety Factor / Limits</th>
                <th style="padding: 10px; text-align: center;">Status</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Toe Pile Axial (Compression)</td>
                <td style="padding: 10px; font-family: monospace;">{P_toe:.2f} kN</td>
                <td style="padding: 10px; font-family: monospace;">{Q_comp_allow:.2f} kN</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">FS = {FS_pile_axial:.2f}</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_toe_axial)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Toe Pile Lateral Shear</td>
                <td style="padding: 10px; font-family: monospace;">{V_max_toe:.2f} kN</td>
                <td style="padding: 10px; font-family: monospace;">{H_allow:.2f} kN</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">FS = {FS_pile_lateral:.2f}</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_toe_lateral)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Toe Pile Structural (Bending-Axial)</td>
                <td style="padding: 10px; font-family: monospace;">M={M_max_toe:.1f} kNm, P={P_toe:.1f} kN</td>
                <td style="padding: 10px; font-family: monospace;">Interaction Envelope</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">-</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_toe_interaction)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Toe Pile Settlement (Axial openpile)</td>
                <td style="padding: 10px; font-family: monospace;">{settlement_toe_pile:.2f} mm</td>
                <td style="padding: 10px; font-family: monospace;">{allowable_pile_settlement_mm:.1f} mm</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">2% Diameter</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_toe_settlement)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Heel Pile Axial ({"Compression" if P_heel >= 0 else "Tension"})</td>
                <td style="padding: 10px; font-family: monospace;">{abs(P_heel):.2f} kN</td>
                <td style="padding: 10px; font-family: monospace;">{Q_comp_allow if P_heel >= 0 else Q_tens_allow:.2f} kN</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">FS = {FS_pile_axial:.2f}</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_heel_axial)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Heel Pile Lateral Shear</td>
                <td style="padding: 10px; font-family: monospace;">{V_max_heel:.2f} kN</td>
                <td style="padding: 10px; font-family: monospace;">{H_allow:.2f} kN</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">FS = {FS_pile_lateral:.2f}</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_heel_lateral)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Heel Pile Structural (Bending-Axial)</td>
                <td style="padding: 10px; font-family: monospace;">M={M_max_heel:.1f} kNm, P={P_heel:.1f} kN</td>
                <td style="padding: 10px; font-family: monospace;">Interaction Envelope</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">-</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_heel_interaction)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eef1f6;">
                <td style="padding: 10px; font-weight: bold;">Heel Pile Settlement (Axial openpile)</td>
                <td style="padding: 10px; font-family: monospace;">{settlement_heel_pile:.2f} mm</td>
                <td style="padding: 10px; font-family: monospace;">{allowable_pile_settlement_mm:.1f} mm</td>
                <td style="padding: 10px; text-align: center; font-family: monospace;">2% Diameter</td>
                <td style="padding: 10px; text-align: center;">{status_span(pile_pass_heel_settlement)}</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

update_splash(splash_placeholder, 95, "Rendering Design Dashboard", "Finalizing figures, rendering verification metrics, and preparing report assets...")

if ssrm_results is not None:
    st.markdown("---")
    with st.container(border=True):
        st.subheader("🧱 Global Stability Analysis (2D SSRM)")
        st.markdown(
            '<div class="section-desc">Global slope stability analysis using 2D Strength Reduction Method (SSRM) with XSLOPE & OpenSees (real wall polygon & Mohr-Coulomb model).</div>',
            unsafe_allow_html=True
        )
        fs_ssrm = ssrm_results.get('FS', None)
        
        # Check status
        is_seismic = (PGA * FPGA) > 0
        req_fs = 1.1 if is_seismic else 1.5
        
        if fs_ssrm is None:
            fs_str = "> 3.000"
            ssrm_pass = True
            fs_delta_str = f"Req >= {req_fs}"
        else:
            fs_str = f"{fs_ssrm:.3f}"
            ssrm_pass = fs_ssrm >= req_fs
            fs_delta_str = f"Req >= {req_fs}"
        
        # Display metric and badge
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.metric(
                label="Factor of Safety (FS)",
                value=fs_str,
                delta=fs_delta_str,
                delta_color="normal" if ssrm_pass else "inverse"
            )
        with col_m2:
            st.markdown(f"<div style='margin-top: 25px;'>Status: {status_span(ssrm_pass)}</div>", unsafe_allow_html=True)
        
        # 10. Slope Geometry & Inputs Plot
        st.markdown("### 1. Slope Geometry & Inputs")
        try:
            import matplotlib
            matplotlib.use('Agg')
            from xslope.plot import plot_inputs
            
            plt.figure(figsize=(10, 4.5))
            plot_inputs(ssrm_results['slope_data'], figsize=(10, 4.5))
            fig_inputs = plt.gcf()
            fig_inputs.tight_layout()
            st.pyplot(fig_inputs)
            plt.close(fig_inputs)
        except Exception as e_inputs:
            st.error(f"Error plotting Slope Geometry & Inputs: {e_inputs}")

        # 11. FEM Data Summary (FEM.Data)
        st.markdown("### 2. FEM Data Summary (FEM.Data)")
        try:
            fem_data_dict = ssrm_results['fem_data']
            n_nodes = len(fem_data_dict['nodes'])
            n_elems = len(fem_data_dict['elements'])
            elem_mats = fem_data_dict['element_materials']
            soil_count = int(np.sum(elem_mats == 1))
            conc_count = int(np.sum(elem_mats == 2))
            
            n_fixed = int(np.sum(fem_data_dict['bc_type'] == 1))
            n_roller = int(np.sum(fem_data_dict['bc_type'] == 2))
            
            u_arr = fem_data_dict.get('u', None)
            has_u = u_arr is not None and len(u_arr) > 0 and u_arr.max() > 0
            u_max_str = f"{u_arr.max():.2f} kPa" if has_u else "None (Dry)"
            
            n_pile_elems = fem_data_dict.get('n_pile_elements', 0)
            pile_info_bullet = f"\n                - **Pile Reinforcement Elements:** `{n_pile_elems}` beam elements (Toe & Heel rows, L = `{L_pile:.1f}`m, D = `{diameter_pile:.2f}`m)" if enable_pile else ""
            
            circles_list = ssrm_results.get('slope_data', {}).get('circles', [])
            if circles_list:
                c1 = circles_list[0]
                circles_info_bullet = f"\n                - **Starting Circles (XSLOPE):** `{len(circles_list)}` circles generated ($X_o={c1['Xo']:.2f}, Y_o={c1['Yo']:.2f}, R={c1['R']:.2f}\\text{{ m}}$)"
            else:
                circles_info_bullet = ""

            col_d1, col_d2 = st.columns([1, 1.3])
            with col_d1:
                st.markdown(f"""
                **Mesh & Domain Parameters:**
                - **Total Mesh Nodes:** `{n_nodes}`
                - **Total 2D Elements (`tri6`):** `{n_elems}`
                - **Soil Zone Elements:** `{soil_count}`
                - **Concrete Wall Elements:** `{conc_count}`{pile_info_bullet}{circles_info_bullet}
                - **Boundary Constraints:** `{n_fixed}` fixed base nodes, `{n_roller}` roller side nodes
                - **Max Pore Pressure ($u_{{max}}$):** `{u_max_str}`
                """)
            with col_d2:
                # Material Summary Table
                mats_list = [
                    {
                        "Material": "Soil",
                        "Model": "Mohr-Coulomb",
                        "γ (kN/m³)": f"{gamma_soil_dry:.1f}",
                        "γ_sat (kN/m³)": f"{gamma_soil_wet:.1f}",
                        "c (kPa)": f"{c_soil:.1f}",
                        "φ (°)": f"{phi:.1f}",
                        "E (kPa)": f"{Es_soil:.1f}",
                        "ν": f"{nu_soil:.2f}",
                        "t_cut": "0.0",
                        "Pore Pressure": "Piezo" if has_u else "None"
                    },
                    {
                        "Material": "Concrete Wall",
                        "Model": "Mohr-Coulomb",
                        "γ (kN/m³)": f"{gamma_c:.1f}",
                        "γ_sat (kN/m³)": "-",
                        "c (kPa)": f"{c_concrete:.1f}",
                        "φ (°)": "40.0",
                        "E (kPa)": "40000.0",
                        "ν": "0.15",
                        "t_cut": "0.0",
                        "Pore Pressure": "None"
                    }
                ]
                if enable_pile:
                    E_pile_val = (Es if pile_material == 'Steel' else Ec) * 1000.0
                    mats_list.append({
                        "Material": f"Pile ({pile_material})",
                        "Model": "Beam / Pile 1D",
                        "γ (kN/m³)": "-",
                        "γ_sat (kN/m³)": "-",
                        "c (kPa)": "-",
                        "φ (°)": "-",
                        "E (kPa)": f"{E_pile_val:.1f}",
                        "ν": "-",
                        "t_cut": "-",
                        "Pore Pressure": "None"
                    })
                mat_df = pd.DataFrame(mats_list)
                st.dataframe(mat_df, hide_index=True, use_container_width=True)

            # FEM Mesh plot with material zones
            from xslope.plot_fem import plot_fem_data
            plt.figure(figsize=(10, 4.5))
            plot_fem_data(ssrm_results['fem_data'], figsize=(10, 4.5), show_bc=True)
            fig_mesh = plt.gcf()
            fig_mesh.tight_layout()
            st.pyplot(fig_mesh)
            plt.close(fig_mesh)
        except Exception as e_data:
            st.error(f"Error displaying FEM.Data: {e_data}")

        # 12. FEM.result (Viscoplastic Shear Strain Failure Plot)
        st.markdown("### 3. FEM Result (FEM.result)")
        try:
            sol = ssrm_results.get('last_solution', None)
            if sol is not None and isinstance(sol, dict):
                from xslope.plot_fem import plot_shear_strain_contours
                fig_ssrm, ax = plt.subplots(1, 1, figsize=(10, 4.5))
                
                plot_shear_strain_contours(ax, ssrm_results['fem_data'], sol, show_mesh=True)
                ax.set_title(f"Viscoplastic Shear Strain (Failure Surface) (FS = {fs_str})", fontsize=11, fontweight='bold')
                
                fig_ssrm.tight_layout()
                try:
                    import os
                    os.makedirs("report_temp", exist_ok=True)
                    fig_ssrm.savefig("report_temp/stability_ssrm.png", dpi=150, bbox_inches='tight')
                except Exception:
                    pass
                st.pyplot(fig_ssrm)
                plt.close(fig_ssrm)
            else:
                st.info("No convergence solution available to plot shear strain contours.")
        except Exception as e_ssrm:
            st.error(f"Error plotting 2D SSRM result: {e_ssrm}")

if splash_placeholder is not None:
    splash_placeholder.empty()
    st.session_state["trigger_splash"] = False

if st.session_state.get("trigger_report_splash", False):
    st.session_state["trigger_report_splash"] = False
    report_splash = st.empty()
    
    update_splash(report_splash, 20, "Exporting Diagrams & High-Res Plots", "Exporting geometry, mesh, and SSRM failure surface plots...")
    os.makedirs("report_temp", exist_ok=True)
    export_report_assets()
    
    pdf_filename = "retaining_wall_report.pdf"
    docx_filename = "retaining_wall_report.docx"
    pdf_path = os.path.join("report_temp", pdf_filename)
    docx_path = os.path.join("report_temp", docx_filename)
    
    update_splash(report_splash, 55, "Generating Word Document (.docx)", "Building calculation tables, checklist, and embedded figures...")
    generate_docx_report(docx_path)
    
    update_splash(report_splash, 85, "Compiling PDF Report (.pdf)", "Generating vector PDF engineering report with ReportLab...")
    generate_pdf_reportlab(pdf_path)
    
    update_splash(report_splash, 100, "Report Generation Complete", "Opening export dialog...")
    import time
    time.sleep(0.2)
    report_splash.empty()
    st.session_state["trigger_report_dialog"] = True

if st.session_state.get("trigger_report_dialog", False):
    st.session_state["trigger_report_dialog"] = False
    show_report_dialog()
        
