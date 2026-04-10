import numpy as np
import OpenGL.GL as GL
import pywavefront
import os

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.transform import identity
from libs.lighting import LightingManager


class Mesh:
    def __init__(self, filename: str):
        self.filename = filename
        self.shader = None
        self.vao = None
        self.vertex_count = 0

    @staticmethod
    def _parse_interleaved(material) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        pywavefront provides a 1D float array in material.vertices, with a per-vertex layout
        described by material.vertex_format (commonly: T2F_N3F_V3F or N3F_V3F).
        """
        verts_1d = np.asarray(material.vertices, dtype=np.float32)
        fmt = getattr(material, "vertex_format", "") or ""

        has_t = "T2F" in fmt
        has_n = "N3F" in fmt

        if has_t and has_n:
            stride = 2 + 3 + 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected vertex stride for format {fmt}: {verts_1d.size} floats")
            v = verts_1d.reshape((-1, stride))
            texcoords = v[:, 0:2]
            normals = v[:, 2:5]
            vertices = v[:, 5:8]
        elif has_n:
            stride = 3 + 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected vertex stride for format {fmt}: {verts_1d.size} floats")
            v = verts_1d.reshape((-1, stride))
            normals = v[:, 0:3]
            vertices = v[:, 3:6]
            texcoords = np.zeros((vertices.shape[0], 2), dtype=np.float32)
        elif has_t:
            stride = 2 + 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected vertex stride for format {fmt}: {verts_1d.size} floats")
            v = verts_1d.reshape((-1, stride))
            texcoords = v[:, 0:2]
            vertices = v[:, 2:5]
            normals = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        else:
            stride = 3
            if verts_1d.size % stride != 0:
                raise ValueError(f"Unexpected vertex stride for format {fmt or 'V3F'}: {verts_1d.size} floats")
            vertices = verts_1d.reshape((-1, 3))
            normals = np.zeros((vertices.shape[0], 3), dtype=np.float32)
            texcoords = np.zeros((vertices.shape[0], 2), dtype=np.float32)

        colors = np.full((vertices.shape[0], 3), (0.6, 0.6, 0.6), dtype=np.float32)
        return vertices, normals, texcoords, colors

    def setup(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vert_path = os.path.join(base_dir, "phong.vert")
        frag_path = os.path.join(base_dir, "phong.frag")
        self.shader = Shader(vert_path, frag_path)

        mesh_path = self.filename
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(base_dir, mesh_path)
        # create_materials=True lets OBJ load even when referenced .mtl is missing.
        scene = pywavefront.Wavefront(
            mesh_path,
            collect_faces=True,
            parse=True,
            create_materials=True,
        )

        vertices_list = []
        normals_list = []
        texcoords_list = []
        colors_list = []

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

        self.vertex_count = int(vertices.shape[0])

        self.vao = VAO()
        self.vao.add_vbo(0, vertices, ncomponents=3)
        self.vao.add_vbo(1, normals, ncomponents=3)
        self.vao.add_vbo(2, texcoords, ncomponents=2)
        self.vao.add_vbo(3, colors, ncomponents=3)

        return self

    def draw(self, projection, view, model=None):
        if model is None:
            model = identity()

        modelview = view @ model

        uma = UManager(self.shader)
        uma.upload_uniform_matrix4fv(projection, "projection", transpose=True)
        uma.upload_uniform_matrix4fv(modelview, "modelview", transpose=True)
        LightingManager(uma).setup_phong()

        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        self.vao.deactivate()

