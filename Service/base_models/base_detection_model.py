from abc import ABC, abstractmethod
import numpy as np

class BaseDetectionModel(ABC):
    @abstractmethod
    def initialize_models(self):
        """Initialize the YOLO and Depth models."""
        pass

    @abstractmethod
    def resizing_image(self, image, width: int, height: int, interpolation_method: int):
        """Resize the input image to the specified dimensions."""
        pass

    @abstractmethod
    def depth_normalize(self, depth_np: np.ndarray):
        """Normalize the depth map for visualization."""
        pass
    @staticmethod
    @abstractmethod
    def ccw(A: tuple, B: tuple, C: tuple):
        """Check if three points are in counter-clockwise order."""
        pass
    @staticmethod
    @abstractmethod
    def line_intersection(line1: list, line2: list):
        """Check if two line segments intersect."""
        pass

    @abstractmethod
    def is_box_intersecting_polygon(self, box: tuple, polygon: np.ndarray):
        """Check if any part of the bounding box intersects with the polygon."""
        pass

    @abstractmethod
    def get_zone(self, center_x: int, mid_width: int, vicinity: bool = False):
        """Classify the object zone as left/right and near/far."""
        pass

    @abstractmethod
    def define_vicinity(self, image: np.ndarray, percentage_vicinity: int, angle_value: int):
        """Define the vicinity region polygon based on percentage and angle."""
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray):
        """
        Process an input frame:
        - Run detection and depth estimation
        - Classify objects as inside or outside the vicinity
        - Annotate and return visualization images
        """
        pass
