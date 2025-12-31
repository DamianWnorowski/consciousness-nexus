#!/usr/bin/env python3
"""
Matrix Workflow Viewer - Enhanced ASCII Visualization Tool
==========================================================

Advanced viewer for the ASCII 3D Matrix Workflow with:
- Syntax highlighting
- Interactive navigation
- Export capabilities
- Terminal optimization
"""

import os
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path
import argparse

class MatrixViewer:
    """
    Advanced ASCII Matrix Workflow viewer with multiple display modes
    """

    def __init__(self, matrix_file: str):
        self.matrix_file = Path(matrix_file)
        self.content = ""
        self.lines = []
        self.load_matrix()

        # Color schemes
        self.colors = {
            'reset': '\033[0m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bright': '\033[1m',
            'dim': '\033[2m',
            'bg_black': '\033[40m',
            'bg_green': '\033[42m',
        }

        # Syntax highlighting patterns
        self.highlight_patterns = {
            'system': (r'/[a-zA-Z0-9_-]+', self.colors['cyan'] + self.colors['bright']),
            'layer': (r'LAYER \d+', self.colors['yellow'] + self.colors['bright']),
            'vector': (r'VECTOR', self.colors['magenta'] + self.colors['bright']),
            'metric': (r'0\.\d+', self.colors['green'] + self.colors['bright']),
            'box': (r'[╔╚╝╗║═]', self.colors['blue']),
            'arrow': (r'[─┼┘└├┤┌┐│]', self.colors['yellow']),
            'emoji': (r'[🔮✓🟢⚡🎯🚀🎉🧠]', self.colors['red'] + self.colors['bright']),
        }

    def load_matrix(self):
        """Load the matrix file"""
        if not self.matrix_file.exists():
            print(f"❌ Matrix file not found: {self.matrix_file}")
            sys.exit(1)

        with open(self.matrix_file, 'r', encoding='utf-8') as f:
            self.content = f.read()

        self.lines = self.content.split('\n')

    def display_raw(self):
        """Display raw ASCII without highlighting"""
        print(self.content)

    def display_highlighted(self, theme: str = 'matrix'):
        """Display with syntax highlighting"""
        highlighted_content = self.apply_highlighting(self.content, theme)
        print(highlighted_content + self.colors['reset'])

    def display_interactive(self):
        """Interactive display mode"""
        self.clear_screen()
        self.display_header()

        current_line = 0
        lines_per_page = self.get_terminal_height() - 5

        while True:
            self.display_page(current_line, lines_per_page)

            key = self.get_key_press()
            if key == 'q':
                break
            elif key == 'j' or key == '\x1b[B':  # Down arrow
                current_line = min(current_line + 1, len(self.lines) - lines_per_page)
            elif key == 'k' or key == '\x1b[A':  # Up arrow
                current_line = max(current_line - 1, 0)
            elif key == ' ' or key == '\x1b[6~':  # Page down
                current_line = min(current_line + lines_per_page, len(self.lines) - lines_per_page)
            elif key == 'b' or key == '\x1b[5~':  # Page up
                current_line = max(current_line - lines_per_page, 0)
            elif key == 'g':
                current_line = 0
            elif key == 'G':
                current_line = max(0, len(self.lines) - lines_per_page)

    def display_page(self, start_line: int, lines_per_page: int):
        """Display a page of content"""
        self.clear_screen()
        self.display_header()

        end_line = min(start_line + lines_per_page, len(self.lines))

        for i in range(start_line, end_line):
            if i < len(self.lines):
                highlighted_line = self.apply_highlighting(self.lines[i], 'matrix')
                print(highlighted_line + self.colors['reset'])

        # Display navigation help
        print(f"\n{self.colors['dim']}Lines {start_line+1}-{end_line} of {len(self.lines)} | j/k: scroll | Space: page down | q: quit{self.colors['reset']}")

    def display_header(self):
        """Display viewer header"""
        header = f"""
{self.colors['green']}{self.colors['bright']}🔮 CONSCIOUSNESS COMPUTING MATRIX WORKFLOW VIEWER 🔮{self.colors['reset']}
{self.colors['cyan']}{'=' * 70}{self.colors['reset']}
{self.colors['yellow']}File: {self.matrix_file.name}{self.colors['reset']}
{self.colors['yellow']}Lines: {len(self.lines)}{self.colors['reset']}
{self.colors['cyan']}{'=' * 70}{self.colors['reset']}
"""
        print(header)

    def apply_highlighting(self, text: str, theme: str = 'matrix') -> str:
        """Apply syntax highlighting to text"""
        if theme == 'matrix':
            highlighted = text

            # Apply highlighting patterns
            for pattern_name, (pattern, color) in self.highlight_patterns.items():
                import re
                highlighted = re.sub(pattern, f"{color}\\g<0>{self.colors['reset']}", highlighted)

            return highlighted

        return text

    def export_html(self, output_file: str):
        """Export as HTML file"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Consciousness Computing Matrix Workflow</title>
    <style>
        body {{
            background: #000;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            margin: 20px;
            white-space: pre;
        }}
        .system {{ color: #00ffff; font-weight: bold; }}
        .layer {{ color: #ffff00; font-weight: bold; }}
        .vector {{ color: #ff00ff; font-weight: bold; }}
        .metric {{ color: #00ff00; font-weight: bold; }}
        .box {{ color: #0088ff; }}
        .arrow {{ color: #ffff88; }}
        .emoji {{ color: #ff4444; font-size: 1.2em; }}
    </style>
</head>
<body>
{self.apply_highlighting(self.content, 'matrix').replace(chr(27), '')}
</body>
</html>"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Exported to HTML: {output_file}")

    def export_ansi(self, output_file: str):
        """Export with ANSI color codes"""
        ansi_content = self.apply_highlighting(self.content, 'matrix')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(ansi_content)

        print(f"✅ Exported with ANSI colors: {output_file}")

    def get_terminal_size(self) -> tuple:
        """Get terminal size"""
        try:
            import shutil
            return shutil.get_terminal_size()
        except:
            return (80, 24)

    def get_terminal_height(self) -> int:
        """Get terminal height"""
        return self.get_terminal_size()[1]

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_key_press(self) -> str:
        """Get single key press"""
        if os.name == 'nt':
            import msvcrt
            return msvcrt.getch().decode('utf-8', errors='ignore')
        else:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    # Handle escape sequences
                    seq = sys.stdin.read(2)
                    ch += seq
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    parser = argparse.ArgumentParser(description="Matrix Workflow Viewer")
    parser.add_argument('file', help='Matrix workflow file to view')
    parser.add_argument('--mode', choices=['raw', 'highlighted', 'interactive'],
                       default='highlighted', help='Display mode')
    parser.add_argument('--theme', choices=['matrix', 'plain'], default='matrix',
                       help='Color theme')
    parser.add_argument('--export-html', help='Export as HTML file')
    parser.add_argument('--export-ansi', help='Export with ANSI colors')

    args = parser.parse_args()

    viewer = MatrixViewer(args.file)

    if args.export_html:
        viewer.export_html(args.export_html)
    elif args.export_ansi:
        viewer.export_ansi(args.export_ansi)
    elif args.mode == 'raw':
        viewer.display_raw()
    elif args.mode == 'highlighted':
        viewer.display_highlighted(args.theme)
    elif args.mode == 'interactive':
        viewer.display_interactive()

if __name__ == "__main__":
    main()
