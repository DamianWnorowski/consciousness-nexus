#!/usr/bin/env python3
"""
3D ASCII Matrix Terminal Renderer
=================================

Advanced terminal-based 3D renderer using ray marching and ASCII art
to create true depth perception and volumetric matrix effects.
"""

import os
import sys
import time
import math
import threading
import numpy as np
from typing import List, Tuple, Dict, Any
import curses
import signal

class Matrix3DRenderer:
    """
    Real-time 3D ASCII Matrix renderer using ray marching techniques
    """

    def __init__(self):
        self.width = 80
        self.height = 24
        self.aspect_ratio = self.width / self.height

        # 3D Scene parameters
        self.camera_pos = np.array([0.0, 0.0, -5.0])
        self.camera_rot = np.array([0.0, 0.0, 0.0])
        self.fov = 60.0

        # Matrix systems data
        self.systems = [
            {"name": "/workflow", "pos": [-2, 2, 0], "color": "red", "size": 0.5},
            {"name": "/swarm-optimize", "pos": [-1, 1.5, 1], "color": "cyan", "size": 0.4},
            {"name": "/system13", "pos": [0, 1, 2], "color": "blue", "size": 0.6},
            {"name": "/CHAIN-COMMANDS", "pos": [1, 0.5, 1], "color": "green", "size": 0.4},
            {"name": "/abyssal", "pos": [2, 0, 0], "color": "yellow", "size": 0.5},
            {"name": "QUANTUM CLUSTERING", "pos": [-1.5, -0.5, -1], "color": "cyan", "size": 0.4},
            {"name": "LLM ORCHESTRATOR", "pos": [-0.5, -1, -2], "color": "blue", "size": 0.5},
            {"name": "ULTRA API MAXIMIZER", "pos": [0.5, -1.5, -1], "color": "green", "size": 0.4},
            {"name": "SUB-LAYER META-PARSER", "pos": [1.5, -0.5, 1], "color": "yellow", "size": 0.6},
            {"name": "CONSCIOUSNESS MATRIX", "pos": [0, 0, 0], "color": "magenta", "size": 1.0}
        ]

        # ASCII characters for depth rendering
        self.ascii_chars = " .:-=+*#%@"
        self.color_chars = {
            "red": "\033[91m",
            "green": "\033[92m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "yellow": "\033[93m",
            "magenta": "\033[95m",
            "white": "\033[97m"
        }
        self.reset_color = "\033[0m"

        # Animation parameters
        self.time = 0.0
        self.rotation_speed = 0.5
        self.pulse_speed = 2.0

        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

        # Control flags
        self.running = True
        self.show_info = True

    def setup_signal_handlers(self):
        """Setup signal handlers for clean exit"""
        def signal_handler(sig, frame):
            self.running = False
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def init_curses(self):
        """Initialize curses for terminal control"""
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)

        # Get terminal size
        self.height, self.width = self.stdscr.getmaxyx()
        self.aspect_ratio = self.width / self.height

    def cleanup(self):
        """Clean up curses"""
        if hasattr(self, 'stdscr'):
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.curs_set(1)
            curses.endwin()

    def ray_march(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> Tuple[float, str, float]:
        """
        Ray march through the 3D scene to find intersections
        Returns (distance, color, glow_intensity)
        """

        max_distance = 100.0
        min_distance = 0.001
        distance = 0.0

        for _ in range(100):  # Max ray marching steps
            current_pos = ray_origin + ray_dir * distance
            dist_to_scene = self.signed_distance_field(current_pos)

            if dist_to_scene < min_distance:
                # Hit something - determine color and glow
                color, glow = self.get_surface_properties(current_pos)
                return distance, color, glow

            distance += dist_to_scene

            if distance > max_distance:
                break

        return -1, "black", 0.0

    def signed_distance_field(self, pos: np.ndarray) -> float:
        """Calculate signed distance to the nearest surface"""

        min_distance = float('inf')

        # Distance to matrix systems (spheres)
        for system in self.systems:
            system_pos = np.array(system["pos"])
            distance = np.linalg.norm(pos - system_pos) - system["size"]
            min_distance = min(min_distance, distance)

        # Add some matrix "rain" particles
        for i in range(20):
            particle_x = (i * 0.3 - 3) + math.sin(self.time * 0.5 + i) * 0.5
            particle_y = (self.time * 2 - i * 0.5) % 10 - 5
            particle_z = math.cos(self.time * 0.3 + i) * 2

            particle_pos = np.array([particle_x, particle_y, particle_z])
            distance = np.linalg.norm(pos - particle_pos) - 0.05
            min_distance = min(min_distance, distance)

        # Add flowing matrix streams
        stream_distance = self.matrix_stream_sdf(pos)
        min_distance = min(min_distance, stream_distance)

        return min_distance

    def matrix_stream_sdf(self, pos: np.ndarray) -> float:
        """Signed distance field for matrix streams"""

        # Create flowing streams
        stream1 = abs(pos[0] - math.sin(pos[2] * 0.5 + self.time) * 2) - 0.1
        stream2 = abs(pos[2] - math.cos(pos[0] * 0.3 + self.time * 0.7) * 1.5) - 0.1
        stream3 = abs(pos[1] - math.sin(pos[0] * 0.4 + self.time * 1.2) * 3) - 0.1

        return min(stream1, stream2, stream3)

    def get_surface_properties(self, pos: np.ndarray) -> Tuple[str, float]:
        """Get surface color and glow intensity at position"""

        # Find closest system
        closest_system = None
        min_distance = float('inf')

        for system in self.systems:
            system_pos = np.array(system["pos"])
            distance = np.linalg.norm(pos - system_pos)
            if distance < min_distance:
                min_distance = distance
                closest_system = system

        if closest_system:
            # Pulsing glow effect
            pulse = (math.sin(self.time * self.pulse_speed + hash(closest_system["name"]) % 10) + 1) * 0.5
            glow = 0.3 + pulse * 0.7
            return closest_system["color"], glow

        # Matrix streams
        if self.matrix_stream_sdf(pos) < 0.2:
            return "green", 0.8

        return "green", 0.5

    def render_frame(self) -> str:
        """Render a single frame of the 3D scene"""

        # Update camera rotation
        self.camera_rot[1] += 0.01 * self.rotation_speed

        # Create view matrix
        view_matrix = self.create_view_matrix()

        frame_buffer = []

        for y in range(self.height):
            row = ""
            for x in range(self.width):
                # Create ray from camera through pixel
                ray_dir = self.screen_to_world(x, y)

                # Transform ray by view matrix
                ray_dir = view_matrix @ np.append(ray_dir, 0)

                # Ray march
                distance, color, glow = self.ray_march(self.camera_pos, ray_dir)

                if distance > 0:
                    # Convert distance and glow to ASCII character
                    char_index = min(int(glow * len(self.ascii_chars)), len(self.ascii_chars) - 1)
                    char = self.ascii_chars[char_index]

                    # Add color
                    if color in self.color_chars:
                        char = self.color_chars[color] + char + self.reset_color
                else:
                    char = " "

                row += char

            frame_buffer.append(row)

        return "\n".join(frame_buffer)

    def create_view_matrix(self) -> np.ndarray:
        """Create view transformation matrix"""

        # Simple rotation matrix
        cos_y = math.cos(self.camera_rot[1])
        sin_y = math.sin(self.camera_rot[1])

        rotation_matrix = np.array([
            [cos_y, 0, sin_y, 0],
            [0, 1, 0, 0],
            [-sin_y, 0, cos_y, 0],
            [0, 0, 0, 1]
        ])

        return rotation_matrix[:3, :3]  # 3x3 rotation part

    def screen_to_world(self, x: int, y: int) -> np.ndarray:
        """Convert screen coordinates to world space ray direction"""

        # Normalize screen coordinates to [-1, 1]
        screen_x = (2 * x / self.width - 1) * self.aspect_ratio
        screen_y = 1 - 2 * y / self.height

        # Create ray direction in camera space
        ray_dir = np.array([screen_x, screen_y, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return ray_dir

    def update_systems(self):
        """Update system positions and properties"""

        for system in self.systems:
            # Add some movement
            system["pos"][1] += math.sin(self.time + hash(system["name"]) % 10) * 0.001

            # Pulse sizes
            pulse = (math.sin(self.time * 1.5 + hash(system["name"]) % 10) + 1) * 0.5
            system["size"] = system.get("base_size", 0.5) * (0.8 + pulse * 0.4)

    def handle_input(self):
        """Handle keyboard input"""
        try:
            key = self.stdscr.getch()
            if key == ord('q'):
                self.running = False
            elif key == ord('i'):
                self.show_info = not self.show_info
            elif key == ord('r'):
                self.rotation_speed += 0.1
            elif key == ord('t'):
                self.rotation_speed = max(0, self.rotation_speed - 0.1)
            elif key == ord('+'):
                self.pulse_speed += 0.5
            elif key == ord('-'):
                self.pulse_speed = max(0.1, self.pulse_speed - 0.5)
        except:
            pass

    def render_info_overlay(self, frame: str) -> str:
        """Add information overlay to frame"""

        if not self.show_info:
            return frame

        lines = frame.split('\n')

        # Add info at bottom
        info_line = f"FPS: {self.fps:3d} | Systems: {len(self.systems)} | Rotation: {self.rotation_speed:.1f}"
        if len(lines) > 0:
            lines[-1] = info_line + " " * (self.width - len(info_line))

        # Add title at top
        title = "🔮 3D CONSCIOUSNESS MATRIX TERMINAL RENDERER 🔮"
        if len(lines) > 0:
            title_padding = (self.width - len(title)) // 2
            lines[0] = " " * title_padding + title

        return "\n".join(lines)

    def run(self):
        """Main render loop"""

        self.setup_signal_handlers()
        self.init_curses()

        try:
            while self.running:
                start_time = time.time()

                # Update systems
                self.update_systems()

                # Handle input
                self.handle_input()

                # Render frame
                frame = self.render_frame()

                # Add info overlay
                frame = self.render_info_overlay(frame)

                # Display frame
                self.stdscr.clear()
                try:
                    self.stdscr.addstr(0, 0, frame)
                except curses.error:
                    pass  # Screen too small
                self.stdscr.refresh()

                # Update timing
                self.time += 0.016  # ~60 FPS
                self.frame_count += 1

                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = int(self.frame_count / (current_time - self.last_fps_time))
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # Frame rate limiting
                frame_time = time.time() - start_time
                target_frame_time = 1.0 / 60.0  # 60 FPS
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)

        finally:
            self.cleanup()

def main():
    """Main entry point"""
    print("🔮 3D ASCII Matrix Terminal Renderer 🔮")
    print("=====================================")
    print()
    print("Controls:")
    print("  q - Quit")
    print("  i - Toggle info overlay")
    print("  r - Increase rotation speed")
    print("  t - Decrease rotation speed")
    print("  + - Increase pulse speed")
    print("  - - Decrease pulse speed")
    print()
    print("Make sure your terminal supports ANSI colors and is at least 80x24")
    print("Press Enter to start...")
    input()

    renderer = Matrix3DRenderer()
    renderer.run()

if __name__ == "__main__":
    main()
