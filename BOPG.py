# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:27:20 2026

@author: RanGe
"""
import numpy as np
import plotly.graph_objects as go

def create_multi_mode_spiral_html():
    # 🎛️ Core parameters
    Lambda = np.pi / 2        # Lambda period node step size
    n_steps = 20              # Total number of periods
    resolution = 100          # Resolution set to 100 as requested

    t_max = n_steps * Lambda
    t_full = np.linspace(0, t_max, resolution)
    z_nodes_full = np.array([i * Lambda for i in range(n_steps + 1)])

    # 🧮 Radius functions r(t) for the three dynamic modes
    def r_constant(t):
        return np.full_like(t, 1.2)           # Constant size: fixed radius = 1.2

    def r_shrink(t):
        return 2.2 * np.exp(-0.11 * t)        # Shrinking (e.g., Mass): contracts centripetally with time/elevation

    def r_expand(t):
        return 0.25 * np.exp(0.09 * t)        # Open / expanding (e.g., Energy): opens up centrifugally with time/elevation

    modes = {
        'constant': {'func': r_constant, 'name': 'Constant Size, e.g. Temporal Base-Frequency Sampling Point', 'color': '#00e5ff'},
        'shrink':   {'func': r_shrink,   'name': 'Shrinking, e.g. Mass',         'color': '#ff1744'},
        'expand':   {'func': r_expand,   'name': 'Open / Expanding, e.g. Energy', 'color': '#76ff03'}
    }

    # 🔒 Globally fixed canvas bounds (covers the maximum envelope of the expanding mode)
    LIMIT_XY = 2.8
    LIMIT_Z = 30.0

    # ---------------------------------------------------------
    # 1. Build the dynamic frame dataset
    # ---------------------------------------------------------
    frames = []
    for k in range(1, len(t_full)):
        t_curr = t_full[k]
        t_c = t_full[:k+1]

        active_mask = z_nodes_full <= t_curr
        z_n_curr = z_nodes_full[active_mask]

        frame_data = [
            # Trace 0: central Lambda axis (Z axis)
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, t_curr]),
            # Trace 1: Lambda key nodes
            go.Scatter3d(x=[0]*len(z_n_curr), y=[0]*len(z_n_curr), z=z_n_curr)
        ]

        # Compute the dynamic geometry for each mode
        for mode_key in ['constant', 'shrink', 'expand']:
            r_func = modes[mode_key]['func']

            # Spiral path
            r_c = r_func(t_c)
            x_c = r_c * np.cos(t_c)
            y_c = r_c * np.sin(t_c)
            z_c = t_c

            # Radial ribs at the nodes
            r_n_curr = r_func(z_n_curr)
            x_n_curr = r_n_curr * np.cos(z_n_curr)
            y_n_curr = r_n_curr * np.sin(z_n_curr)

            rib_x, rib_y, rib_z = [], [], []
            for xi, yi, zi in zip(x_n_curr, y_n_curr, z_n_curr):
                rib_x.extend([0, xi, None])
                rib_y.extend([0, yi, None])
                rib_z.extend([zi, zi, None])

            # Leading end point
            r_head = r_func(t_curr)
            x_head = r_head * np.cos(t_curr)
            y_head = r_head * np.sin(t_curr)

            frame_data.extend([
                go.Scatter3d(x=rib_x, y=rib_y, z=rib_z),
                go.Scatter3d(x=x_c, y=y_c, z=z_c),
                go.Scatter3d(x=[x_head], y=[y_head], z=[t_curr])
            ])

        frames.append(go.Frame(data=frame_data, name=f"f_{k}"))

    # ---------------------------------------------------------
    # 2. Build the initial traces (corresponding to the k = 1 state)
    # ---------------------------------------------------------
    init_k = 1
    t_curr_init = t_full[init_k]
    t_c_init = t_full[:init_k+1]
    z_n_init = z_nodes_full[z_nodes_full <= t_curr_init]

    initial_traces = [
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, t_curr_init], mode='lines',
                     line=dict(color='#ff4081', width=5), name='Central Spine (Z Axis)'),
        go.Scatter3d(x=[0]*len(z_n_init), y=[0]*len(z_n_init), z=z_n_init, mode='markers',
                     marker=dict(size=6, color='#ffea00'), name='Lambda Nodes')
    ]

    for mode_key in ['constant', 'shrink', 'expand']:
        m = modes[mode_key]
        r_func = m['func']

        r_c = r_func(t_c_init)
        x_c, y_c, z_c = r_c * np.cos(t_c_init), r_c * np.sin(t_c_init), t_c_init

        r_n = r_func(z_n_init)
        x_n, y_n = r_n * np.cos(z_n_init), r_n * np.sin(z_n_init)
        rib_x, rib_y, rib_z = [], [], []
        for xi, yi, zi in zip(x_n, y_n, z_n_init):
            rib_x.extend([0, xi, None])
            rib_y.extend([0, yi, None])
            rib_z.extend([zi, zi, None])

        r_h = r_func(t_curr_init)
        x_h, y_h = r_h * np.cos(t_curr_init), r_h * np.sin(t_curr_init)

        # By default only the "Constant Size" mode is shown; the others stay hidden
        is_visible = (mode_key == 'constant')

        initial_traces.extend([
            go.Scatter3d(x=rib_x, y=rib_y, z=rib_z, mode='lines',
                         line=dict(color='#76ff03', width=2, dash='dot'),
                         name=f'Radius Ribs ({m["name"]})', visible=is_visible),
            go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='lines',
                         line=dict(color=m['color'], width=5),
                         name=f'Spiral ({m["name"]})', visible=is_visible),
            go.Scatter3d(x=[x_h], y=[y_h], z=[t_curr_init], mode='markers',
                         marker=dict(size=10, color='#ff1744', symbol='diamond'),
                         name=f'Head ({m["name"]})', visible=is_visible)
        ])

    fig = go.Figure(data=initial_traces, frames=frames)

    # ---------------------------------------------------------
    # 3. Interactive menus and control settings
    # ---------------------------------------------------------
    fig.update_layout(
        title="<b>Lambda Main-Axis Interactive Open Spiral Dynamics</b>",
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        uirevision='fixed_state',
        scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', backgroundcolor='#0d1117', range=[-LIMIT_XY, LIMIT_XY], autorange=False),
            yaxis=dict(title='Y', backgroundcolor='#0d1117', range=[-LIMIT_XY, LIMIT_XY], autorange=False),
            zaxis=dict(title='Z (Lambda Axis)', backgroundcolor='#0d1117', range=[0, LIMIT_Z], autorange=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.1))
        ),
        updatemenus=[
            # 🔘 Mode-switch dropdown (top left)
            dict(
                type="dropdown",
                direction="down",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                active=0,
                showactive=True,
                bgcolor='#21262d',
                font=dict(color='white', size=12),
                buttons=[
                    dict(
                        label="🔄 Mode: Constant Size (Constant), e.g. Temporal Base-Frequency Sampling Point",
                        method="restyle",
                        args=[{"visible": [True, True,  True, True, True,  False, False, False,  False, False, False]}]
                    ),
                    dict(
                        label="📉 Mode: Shrinking Spiral (Shrinking), e.g. Mass",
                        method="restyle",
                        args=[{"visible": [True, True,  False, False, False,  True, True, True,  False, False, False]}]
                    ),
                    dict(
                        label="📈 Mode: Open Expanding Spiral (Expanding), e.g. Energy",
                        method="restyle",
                        args=[{"visible": [True, True,  False, False, False,  False, False, False,  True, True, True]}]
                    )
                ]
            ),
            # ⏯️ Play / pause buttons (bottom left)
            dict(
                type="buttons",
                showactive=False,
                x=0.02, y=0.02,
                xanchor="left", yanchor="bottom",
                buttons=[
                    dict(label="► Play Dynamic", method="animate",
                         args=[None, dict(frame=dict(duration=35, redraw=True), fromcurrent=True)]),
                    dict(label="❚❚ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ],
                font=dict(color='white'),
                bgcolor='#21262d'
            )
        ],
        legend=dict(font=dict(color='white'), bgcolor='#161b22')
    )

    file_name = "BOPG-spiral_interactive_dynamics.html"
    fig.write_html(file_name)
    print(f"-> Successfully generated the open-spiral interactive dynamics canvas: {file_name}")

if __name__ == "__main__":
    create_multi_mode_spiral_html()