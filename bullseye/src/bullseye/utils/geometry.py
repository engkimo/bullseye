from __future__ import annotations

import numpy as np


def is_vertical(quad, thresh_aspect: float = 2.0) -> bool:
    """Heuristically judge if a quadrilateral word box is vertical by aspect ratio."""
    quad = np.array(quad)
    width = np.linalg.norm(quad[0] - quad[1])
    height = np.linalg.norm(quad[1] - quad[2])
    return bool(height > width * thresh_aspect)


def is_noise(quad, thresh: float = 15.0) -> bool:
    """Treat extremely small boxes as noise (in pixels)."""
    quad = np.array(quad)
    width = np.linalg.norm(quad[0] - quad[1])
    height = np.linalg.norm(quad[1] - quad[2])
    return bool(width < thresh or height < thresh)

