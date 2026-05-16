from typing import Deque, Optional, Tuple
from collections import deque

import cv2
import numpy as np


class DrawingCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_point: Optional[Tuple[int, int]] = None
        self.draw_color = (0, 255, 0)
        self.thickness = 5
        self.tool = "pen"
        self.eraser_radius = 20
        self._history: Deque[np.ndarray] = deque(maxlen=10)
        self._redo: Deque[np.ndarray] = deque(maxlen=10)
        self._stroke_active = False

    def draw(self, point: Tuple[int, int]) -> None:
        if not self._stroke_active:
            self._push_history()
            self._stroke_active = True
        if self.last_point is not None:
            color, thickness = self._tool_style()
            cv2.line(self.canvas, self.last_point, point, color, thickness)
        self.last_point = point

    def erase(self, point: Tuple[int, int]) -> None:
        if not self._stroke_active:
            self._push_history()
            self._stroke_active = True
        cv2.circle(self.canvas, point, self.eraser_radius, (0, 0, 0), -1)
        self.last_point = None

    def clear(self) -> None:
        self._push_history()
        self.canvas[:] = 0
        self.last_point = None
        self._stroke_active = False

    def reset_last_point(self) -> None:
        self.last_point = None
        self._stroke_active = False

    def get_canvas(self) -> np.ndarray:
        return self.canvas.copy()

    def set_tool(self, tool: str) -> None:
        self.tool = tool

    def set_color(self, bgr: Tuple[int, int, int]) -> None:
        self.draw_color = bgr

    def set_thickness(self, value: int) -> None:
        self.thickness = max(1, min(30, value))
        self.eraser_radius = max(10, self.thickness * 3)

    def increase_thickness(self) -> None:
        self.set_thickness(self.thickness + 1)

    def decrease_thickness(self) -> None:
        self.set_thickness(self.thickness - 1)

    def undo(self) -> None:
        if not self._history:
            return
        self._redo.append(self.canvas.copy())
        self.canvas = self._history.pop()

    def redo(self) -> None:
        if not self._redo:
            return
        self._history.append(self.canvas.copy())
        self.canvas = self._redo.pop()

    def _push_history(self) -> None:
        self._history.append(self.canvas.copy())
        self._redo.clear()

    def _tool_style(self) -> Tuple[Tuple[int, int, int], int]:
        if self.tool == "pencil":
            color = tuple(int(c * 0.6) for c in self.draw_color)
            thickness = max(1, self.thickness - 2)
            return color, thickness
        if self.tool == "marker":
            thickness = self.thickness + 4
            return self.draw_color, thickness
        return self.draw_color, self.thickness
