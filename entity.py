import numpy as np
import OpenGL.GL as GL
import pywavefront
import os

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.transform import identity
from libs.lighting import LightingManager

from mesh import Mesh

#### Từ Mesh upload đây là nơi sẽ quản lý từng đối tượng
class Entity():
    def __init__(self):
        pass 
    
    def set_transform(self):
        pass 


#### Bộ tổng hợp Entity để tạo thành xử lý chung cho toàn màn hình
class Scene():
    def __init__(self):
        pass 
    
    def add_entity(self):
        pass
     
        