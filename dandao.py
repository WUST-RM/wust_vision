import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

g = 9.81  # Gravity

def calculate_trajectory(angle_deg, v0, distance, height_diff, drag_coefficient=0, mass=1):
    """
    Calculate trajectory with air resistance using Euler integration.
    """
    angle_rad = math.radians(angle_deg)
    dt = 0.01
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    x, y = 0, 0
    xs, ys = [x], [y]

    while y >= -0.1 and x <= distance * 1.5:
        v = math.sqrt(vx**2 + vy**2)
        ax = -drag_coefficient * v * vx / mass
        ay = -g - (drag_coefficient * v * vy / mass)

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        ys.append(y)
        xs.append(x)
        if x >= distance and abs(y - height_diff) < 0.1:
            break
    return np.array(xs), np.array(ys)

def calculate_trajectory_no_drag(angle_deg, distance, height_diff, initial_velocity):
    angle_rad = math.radians(angle_deg)
    vx = initial_velocity * math.cos(angle_rad)
    vy = initial_velocity * math.sin(angle_rad)

    flight_time = distance / vx
    time_points = np.linspace(0, flight_time, 200)
    x_points = vx * time_points
    y_points = vy * time_points - 0.5 * g * time_points ** 2

    return x_points, y_points

def calculate_angles(distance, height_diff, v0):
    if distance == 0:
        if height_diff > 0:
            return [90.0]
        elif height_diff < 0:
            return [-90.0]
        else:
            return [0.0]

    v2 = v0 ** 2
    disc = v2 ** 2 - g * (g * distance ** 2 + 2 * height_diff * v2)
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    angle1 = math.atan((v2 + sqrt_disc) / (g * distance))
    angle2 = math.atan((v2 - sqrt_disc) / (g * distance))
    return sorted([math.degrees(angle1), math.degrees(angle2)])

class TrajectoryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trajectory Simulator (with Air Resistance)")

      
        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10)

        ttk.Label(frame, text="Horizontal Distance (m):").grid(row=0, column=0, sticky="w")
        self.distance_var = tk.DoubleVar(value=50)
        ttk.Entry(frame, textvariable=self.distance_var, width=10).grid(row=0, column=1)

        ttk.Label(frame, text="Height Difference (m):").grid(row=1, column=0, sticky="w")
        self.height_var = tk.DoubleVar(value=0)
        ttk.Entry(frame, textvariable=self.height_var, width=10).grid(row=1, column=1)

        ttk.Label(frame, text="Initial Velocity (m/s):").grid(row=2, column=0, sticky="w")
        self.velocity_var = tk.DoubleVar(value=30)
        ttk.Entry(frame, textvariable=self.velocity_var, width=10).grid(row=2, column=1)

        ttk.Label(frame, text="Drag Coefficient (kg/m):").grid(row=3, column=0, sticky="w")
        self.drag_var = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.drag_var, width=10).grid(row=3, column=1)

        ttk.Button(frame, text="Compute & Plot", command=self.update_plot).grid(row=4, column=0, columnspan=2, pady=8)


        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack()

        self.update_plot()

    def update_plot(self):
        distance = self.distance_var.get()
        height_diff = self.height_var.get()
        velocity = self.velocity_var.get()
        drag = self.drag_var.get()

        angles = calculate_angles(distance, height_diff, velocity)
        if angles is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No valid solution. Please adjust parameters.',
                         ha='center', va='center', fontsize=14, color='red', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        self.ax.clear()
        self.ax.plot(0, 0, 'go', label='Launch Point')
        self.ax.plot(distance, height_diff, 'ro', label='Target Point')
        self.ax.set_xlabel("Horizontal Distance (m)")
        self.ax.set_ylabel("Height (m)")
        self.ax.grid(True)

        if drag > 0:
            x, y = calculate_trajectory(angles[0], velocity, distance, height_diff, drag_coefficient=drag)
            self.ax.plot(x, y, 'b-', label=f'With Drag: {angles[0]:.1f}°')
            all_y = y
        else:
            x1, y1 = calculate_trajectory_no_drag(angles[0], distance, height_diff, velocity)
            x2, y2 = calculate_trajectory_no_drag(angles[1], distance, height_diff, velocity)
            self.ax.plot(x1, y1, 'b-', label=f'Low Angle: {angles[0]:.1f}°')
            self.ax.plot(x2, y2, 'g-', label=f'High Angle: {angles[1]:.1f}°')
            all_y = np.concatenate([y1, y2])

    
        self.ax.set_xlim(left=0)
        y_min = min(all_y.min(), 0)
        y_margin = (all_y.max() - y_min) * 0.1
        self.ax.set_ylim(y_min - y_margin, all_y.max() + y_margin)

        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    try:
        app = TrajectoryApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\nProgram exited via Ctrl+C.")
