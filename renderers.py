import numpy as np
import OpenGL.GL as GL
import pywavefront
import os

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.transform import identity
from libs.lighting import LightingManager

from annotations import BBoxCalculator, BBoxVisualizer, DatasetExporter


##### đây là bộ giúp render ra ảnh màu RGB theo phong shader
class RGBRenderer():
    def __init__():
        pass
    
#### đây là bộ tạo mặt nạ dựa vào mask shader tự định nghĩa để tách biệt các instance ra
class MaskRenderer():
    def __init__():
        pass
    
#### bộ tạo depth dựa vào depth shader tự định nghĩa
class DepthRenderer():
    def __init__():
        pass
        
### Bộ quản lý chức năng tổng --- 
#####lưu ý tách xử lý bounding box và capture lại frame trên 1 annotation.py
class RendererManager():
    def __init__(self):
        pass
        DatasetExporter()
    
    def set_mode(self, mode):
        if mode == "RGB":
            pass
        elif mode == "DEPTH":
            pass
        elif mode == "MASK":
            pass
        
    def toggle_bbox_visualization():
        pass
    
    def export_current_frame():
        DatasetExporter()
        pass