# mesh.py
import os
from typing import Tuple, List

import numpy as np
import OpenGL.GL as GL
import pywavefront

from libs.buffer import VAO, UManager
from libs.transform import identity


class Mesh:
    """
    Low-level mesh container:
    - Load OBJ via pywavefront
    - Build VAO/VBO for position/normal/uv/color
    - Keep a CPU copy of vertices for annotation pipeline (self.vertices: Nx3)
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.vao = None
        self.vertex_count = 0

        # Required by annotation stage (CPU projection for bbox)
        self.vertices: np.ndarray | None = None

        # Local AABB cache
        self.local_aabb_min: np.ndarray | None = None
        self.local_aabb_max: np.ndarray | None = None

    @staticmethod
    def _parse_interleaved(material) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        pywavefront stores interleaved vertex data in material.vertices.
        Common formats:
        - T2F_N3F_V3F
        - N3F_V3F
        - T2F_V3F
        - V3F
        """
        verts_1d = np.asarray(material.vertices, dtype=np.float32)
        fmt = (getattr(material, "vertex_format", "") or "").upper()

        has_t = "T2F" in fmt
        has_n = "N3F" in fmt

        if has_t and has_n:
            stride = 2 + 3 + 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected stride for {fmt}: {verts_1d.size} floats")
            arr = verts_1d.reshape((-1, stride))
            texcoords = arr[:, 0:2]
            normals = arr[:, 2:5]
            vertices = arr[:, 5:8]
        elif has_n:
            stride = 3 + 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected stride for {fmt}: {verts_1d.size} floats")
            arr = verts_1d.reshape((-1, stride))
            normals = arr[:, 0:3]
            vertices = arr[:, 3:6]
            texcoords = np.zeros((vertices.shape[0], 2), dtype=np.float32)
        elif has_t:
            stride = 2 + 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected stride for {fmt}: {verts_1d.size} floats")
            arr = verts_1d.reshape((-1, stride))
            texcoords = arr[:, 0:2]
            vertices = arr[:, 2:5]
            normals = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        else:
            stride = 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected stride for {fmt or 'V3F'}: {verts_1d.size} floats")
            vertices = verts_1d.reshape((-1, 3))
            normals = np.zeros((vertices.shape[0], 3), dtype=np.float32)
            texcoords = np.zeros((vertices.shape[0], 2), dtype=np.float32)

        # Try to derive color from material diffuse, fallback gray
        diffuse = getattr(material, "diffuse", None)
        if diffuse is not None and len(diffuse) >= 3:
            color = np.array(diffuse[:3], dtype=np.float32)
        else:
            color = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        colors = np.tile(color[None, :], (vertices.shape[0], 1)).astype(np.float32)

        return vertices, normals, texcoords, colors

    def compute_local_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute local-space AABB from CPU vertices.
        Must stay purely geometry-side (no shader/lighting logic).
        """
        if self.vertices is None or self.vertices.size == 0:
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero

        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        return min_coords.astype(np.float32), max_coords.astype(np.float32)

    def setup(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_path = self.filename
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(base_dir, mesh_path)

        scene = pywavefront.Wavefront(
            mesh_path,
            collect_faces=True,
            parse=True,
            create_materials=True,
        )

        vertices_list: List[np.ndarray] = []
        normals_list: List[np.ndarray] = []
        texcoords_list: List[np.ndarray] = []
        colors_list: List[np.ndarray] = []

        for material in scene.materials.values():
            if not getattr(material, "vertices", None):
                continue
            v, n, t, c = self._parse_interleaved(material)
            vertices_list.append(v)
            normals_list.append(n)
            texcoords_list.append(t)
            colors_list.append(c)

        if not vertices_list:
            raise ValueError(f"No vertices found in OBJ: {self.filename}")

        vertices = np.vstack(vertices_list).astype(np.float32, copy=False)
        normals = np.vstack(normals_list).astype(np.float32, copy=False)
        texcoords = np.vstack(texcoords_list).astype(np.float32, copy=False)
        colors = np.vstack(colors_list).astype(np.float32, copy=False)

        self.vertices = vertices
        self.vertex_count = int(vertices.shape[0])
        self.local_aabb_min, self.local_aabb_max = self.compute_local_aabb()

        self.vao = VAO()
        self.vao.add_vbo(0, vertices, ncomponents=3)   # position
        self.vao.add_vbo(1, normals, ncomponents=3)    # normal
        self.vao.add_vbo(2, texcoords, ncomponents=2)  # uv
        self.vao.add_vbo(3, colors, ncomponents=3)     # color

        return self

    def draw(self, projection: np.ndarray, view: np.ndarray, model: np.ndarray | None, shader):
        """
        Generic draw using an external shader (works for RGB/Mask/Depth passes).
        """
        if model is None:
            model = identity()

        modelview = view @ model

        uma = UManager(shader)
        uma.upload_uniform_matrix4fv(projection, "projection", transpose=True)
        uma.upload_uniform_matrix4fv(modelview, "modelview", transpose=True)

        # Optional for shaders that may use world model matrix (safe if uniform not found).
        uma.upload_uniform_matrix4fv(model, "model", transpose=True)

        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        self.vao.deactivate()