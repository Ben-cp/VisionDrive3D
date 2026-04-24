import os
import json
import struct
from typing import Tuple, List

import numpy as np
import OpenGL.GL as GL
import pywavefront
from PIL import Image
from io import BytesIO

from libs.buffer import VAO, UManager
from libs.transform import identity


class Mesh:
    """
    Low-level mesh container:
    - Load OBJ via pywavefront
    - Build VAO/VBO for position/normal/uv/color
    - Keep CPU vertices for annotation pipeline (self.vertices: Nx3)
    - Optionally load diffuse texture from MTL map_Kd
    """

    def __init__(self, filename: str, max_submesh_extent: float | None = None):
        self.filename = filename
        self.max_submesh_extent = max_submesh_extent
        self.vao = None
        self.vertex_count = 0
        self.submeshes: List[dict] = []

        # Required by annotation stage (CPU projection for bbox)
        self.vertices: np.ndarray | None = None

        # Local AABB cache
        self.local_aabb_min: np.ndarray | None = None
        self.local_aabb_max: np.ndarray | None = None

        # Optional diffuse texture loaded from MTL map_Kd
        self.texture_id: int | None = None
        self.has_texture: bool = False

    _GLTF_COMPONENT_DTYPES = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }

    _GLTF_TYPE_COUNTS = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

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

        diffuse = getattr(material, "diffuse", None)
        if diffuse is not None and len(diffuse) >= 3:
            color = np.array(diffuse[:3], dtype=np.float32)
        else:
            color = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        colors = np.tile(color[None, :], (vertices.shape[0], 1)).astype(np.float32)

        return vertices, normals, texcoords, colors

    @staticmethod
    def _resolve_mtl_path(obj_path: str) -> str | None:
        """Resolve first mtllib entry from OBJ file."""
        if not os.path.exists(obj_path):
            return None

        obj_dir = os.path.dirname(obj_path)
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.lower().startswith("mtllib "):
                        # Keep the rest of the line intact (mtl names can contain spaces)
                        mtl_rel = s[7:].strip()
                        mtl_path = os.path.join(obj_dir, mtl_rel)
                        if os.path.exists(mtl_path):
                            return mtl_path
                        break
        except OSError:
            return None

        return None

    @staticmethod
    def _parse_map_kd_from_mtl(mtl_path: str) -> str | None:
        """
        Parse first resolvable map_Kd texture path from MTL.

        Many assets export absolute Windows paths in MTL. If these paths are
        invalid on current machine, fallback to local textures folders.
        """
        if not mtl_path or not os.path.exists(mtl_path):
            return None

        mtl_dir = os.path.dirname(mtl_path)

        def _resolve_texture_candidate(raw_tex: str) -> str | None:
            raw_tex = raw_tex.strip().strip('"').replace('\\', os.sep)
            if not raw_tex:
                return None

            candidates: List[str] = []

            def _try_push(path_str: str):
                if path_str and path_str not in candidates:
                    candidates.append(path_str)

            # 1) As-is absolute path from exporter
            _try_push(raw_tex)

            # 2) Relative to mtl directory
            _try_push(os.path.join(mtl_dir, raw_tex))

            # 3) Local fallback by basename in nearby texture folders
            tex_name = os.path.basename(raw_tex)
            _try_push(os.path.join(mtl_dir, tex_name))
            _try_push(os.path.join(mtl_dir, "textures", tex_name))
            _try_push(os.path.join(os.path.dirname(mtl_dir), "textures", tex_name))

            for c in candidates:
                if os.path.exists(c):
                    return c
            return None

        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.lower().startswith("map_kd "):
                        tex_rel = s[7:].strip()
                        # map_Kd may contain options (e.g. -blendu on). Keep last token.
                        if tex_rel.startswith("-") and " " in tex_rel:
                            tex_rel = tex_rel.split()[-1]
                        tex_path = _resolve_texture_candidate(tex_rel)
                        if tex_path:
                            return tex_path
        except OSError:
            return None

        return None

    @staticmethod
    def _parse_mtl_materials(mtl_path: str) -> dict[str, dict]:
        """
        Parse per-material diffuse color, emissive color and map_Kd from MTL.
        """
        if not mtl_path or not os.path.exists(mtl_path):
            return {}

        materials: dict[str, dict] = {}
        current_name: str | None = None
        mtl_dir = os.path.dirname(mtl_path)

        def _resolve_texture_candidate(raw_tex: str) -> str | None:
            raw_tex = raw_tex.strip().strip('"').replace('\\', os.sep)
            if not raw_tex:
                return None

            candidates: List[str] = []

            def _try_push(path_str: str):
                if path_str and path_str not in candidates:
                    candidates.append(path_str)

            _try_push(raw_tex)
            _try_push(os.path.join(mtl_dir, raw_tex))

            tex_name = os.path.basename(raw_tex)
            _try_push(os.path.join(mtl_dir, tex_name))
            _try_push(os.path.join(mtl_dir, "textures", tex_name))
            _try_push(os.path.join(os.path.dirname(mtl_dir), "textures", tex_name))

            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return None

        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue

                    if s.lower().startswith("newmtl "):
                        current_name = s[7:].strip()
                        materials[current_name] = {
                            "diffuse": np.array([0.6, 0.6, 0.6], dtype=np.float32),
                            "emissive": np.zeros(3, dtype=np.float32),
                            "texture_path": None,
                        }
                        continue

                    if current_name is None:
                        continue

                    if s.lower().startswith("kd "):
                        parts = s.split()
                        if len(parts) >= 4:
                            try:
                                materials[current_name]["diffuse"] = np.array(
                                    [float(parts[1]), float(parts[2]), float(parts[3])],
                                    dtype=np.float32,
                                )
                            except ValueError:
                                pass
                        continue

                    if s.lower().startswith("ke "):
                        parts = s.split()
                        if len(parts) >= 4:
                            try:
                                materials[current_name]["emissive"] = np.array(
                                    [float(parts[1]), float(parts[2]), float(parts[3])],
                                    dtype=np.float32,
                                )
                            except ValueError:
                                pass
                        continue

                    if s.lower().startswith("map_kd "):
                        tex_rel = s[7:].strip()
                        if tex_rel.startswith("-") and " " in tex_rel:
                            tex_rel = tex_rel.split()[-1]
                        materials[current_name]["texture_path"] = _resolve_texture_candidate(tex_rel)
        except OSError:
            return {}

        return materials

    @staticmethod
    def _create_gl_texture(image_path: str) -> int | None:
        """Create GL_TEXTURE_2D from image using PIL."""
        if not image_path or not os.path.exists(image_path):
            return None

        try:
            img = Image.open(image_path).convert("RGBA")
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(img, dtype=np.uint8)

            tex_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA,
                img.width,
                img.height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                img_data,
            )
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            return int(tex_id)
        except Exception:
            return None

    @staticmethod
    def _create_gl_texture_from_bytes(image_bytes: bytes) -> int | None:
        """Create GL texture from in-memory encoded image bytes."""
        if not image_bytes:
            return None

        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGBA")
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(img, dtype=np.uint8)

            tex_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA,
                img.width,
                img.height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                img_data,
            )
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            return int(tex_id)
        except Exception:
            return None

    @staticmethod
    def _quaternion_to_matrix(quat: list[float]) -> np.ndarray:
        x, y, z, w = [float(v) for v in quat]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        return mat

    @classmethod
    def _gltf_node_matrix(cls, node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float32).reshape((4, 4), order="F")

        translation = np.eye(4, dtype=np.float32)
        translation[:3, 3] = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)

        rotation = cls._quaternion_to_matrix(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))

        scale = np.eye(4, dtype=np.float32)
        scale_values = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        scale[0, 0], scale[1, 1], scale[2, 2] = scale_values

        return translation @ rotation @ scale

    @classmethod
    def _read_gltf_accessor(cls, gltf: dict, accessor_idx: int, buffers: list[bytes]) -> np.ndarray:
        accessor = gltf["accessors"][accessor_idx]
        component_type = accessor["componentType"]
        dtype = cls._GLTF_COMPONENT_DTYPES.get(component_type)
        if dtype is None:
            raise ValueError(f"Unsupported glTF accessor componentType: {component_type}")

        accessor_type = accessor["type"]
        component_count = cls._GLTF_TYPE_COUNTS.get(accessor_type)
        if component_count is None:
            raise ValueError(f"Unsupported glTF accessor type: {accessor_type}")

        count = int(accessor["count"])
        if "bufferView" not in accessor:
            return np.zeros((count, component_count), dtype=dtype)

        buffer_view = gltf["bufferViews"][accessor["bufferView"]]
        raw_buffer = buffers[buffer_view.get("buffer", 0)]
        accessor_offset = int(accessor.get("byteOffset", 0))
        view_offset = int(buffer_view.get("byteOffset", 0))
        byte_offset = view_offset + accessor_offset
        stride = int(buffer_view.get("byteStride", 0))
        elem_nbytes = np.dtype(dtype).itemsize * component_count

        if stride in (0, elem_nbytes):
            flat = np.frombuffer(raw_buffer, dtype=dtype, count=count * component_count, offset=byte_offset)
            return flat.reshape((count, component_count))

        out = np.empty((count, component_count), dtype=dtype)
        for i in range(count):
            start = byte_offset + i * stride
            out[i] = np.frombuffer(raw_buffer, dtype=dtype, count=component_count, offset=start)
        return out

    @staticmethod
    def _expand_indexed_primitive(
        positions: np.ndarray,
        normals: np.ndarray | None,
        texcoords: np.ndarray | None,
        indices: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if indices is None:
            expanded_positions = positions.astype(np.float32, copy=False)
            expanded_normals = (
                normals.astype(np.float32, copy=False)
                if normals is not None
                else np.zeros_like(expanded_positions, dtype=np.float32)
            )
            expanded_texcoords = (
                texcoords.astype(np.float32, copy=False)
                if texcoords is not None
                else np.zeros((expanded_positions.shape[0], 2), dtype=np.float32)
            )
            return expanded_positions, expanded_normals, expanded_texcoords

        flat_indices = indices.reshape(-1).astype(np.int64, copy=False)
        expanded_positions = positions[flat_indices].astype(np.float32, copy=False)
        expanded_normals = (
            normals[flat_indices].astype(np.float32, copy=False)
            if normals is not None
            else np.zeros_like(expanded_positions, dtype=np.float32)
        )
        expanded_texcoords = (
            texcoords[flat_indices].astype(np.float32, copy=False)
            if texcoords is not None
            else np.zeros((expanded_positions.shape[0], 2), dtype=np.float32)
        )
        return expanded_positions, expanded_normals, expanded_texcoords

    @staticmethod
    def _transform_positions(positions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        ones = np.ones((positions.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack((positions.astype(np.float32, copy=False), ones))
        transformed = (matrix @ homogeneous.T).T
        return transformed[:, :3].astype(np.float32, copy=False)

    @staticmethod
    def _transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if normals.size == 0:
            return normals.astype(np.float32, copy=False)

        normal_matrix = np.linalg.inv(matrix[:3, :3]).T.astype(np.float32, copy=False)
        transformed = normals.astype(np.float32, copy=False) @ normal_matrix.T
        lengths = np.linalg.norm(transformed, axis=1, keepdims=True)
        lengths[lengths < 1e-8] = 1.0
        return (transformed / lengths).astype(np.float32, copy=False)

    @classmethod
    def _load_obj_submeshes(cls, mesh_path: str) -> tuple[list[dict], np.ndarray]:
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
        submeshes: List[dict] = []

        mtl_path = cls._resolve_mtl_path(mesh_path)
        mtl_materials = cls._parse_mtl_materials(mtl_path) if mtl_path else {}
        texture_cache: dict[str, int] = {}

        for material_name, material in scene.materials.items():
            if not getattr(material, "vertices", None):
                continue
            v, n, t, c = cls._parse_interleaved(material)
            vertices_list.append(v)
            normals_list.append(n)
            texcoords_list.append(t)
            colors_list.append(c)

            material_info = mtl_materials.get(material_name, {})
            texture_path = material_info.get("texture_path")
            if texture_path:
                tex_id = texture_cache.get(texture_path)
                if tex_id is None:
                    tex_id = cls._create_gl_texture(texture_path)
                    if tex_id is not None:
                        texture_cache[texture_path] = tex_id
            else:
                tex_id = None

            base_color = material_info.get("diffuse")
            if base_color is None:
                diffuse = getattr(material, "diffuse", None)
                if diffuse is not None and len(diffuse) >= 3:
                    base_color = np.array(diffuse[:3], dtype=np.float32)
                else:
                    base_color = np.array([0.6, 0.6, 0.6], dtype=np.float32)

            emissive_color = np.asarray(material_info.get("emissive", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            emissive_strength = float(np.linalg.norm(emissive_color))

            submesh_vao = VAO()
            submesh_vao.add_vbo(0, v, ncomponents=3)
            submesh_vao.add_vbo(1, n, ncomponents=3)
            submesh_vao.add_vbo(2, t, ncomponents=2)
            submesh_vao.add_vbo(3, c, ncomponents=3)
            submeshes.append(
                {
                    "vao": submesh_vao,
                    "vertex_count": int(v.shape[0]),
                    "texture_id": tex_id,
                    "has_texture": tex_id is not None,
                    "base_color": np.asarray(base_color, dtype=np.float32),
                    "material_name": material_name,
                    "emissive_color": emissive_color,
                    "emissive_strength": emissive_strength,
                    "vertices": v.astype(np.float32, copy=False),
                }
            )

        if not vertices_list:
            raise ValueError(f"No vertices found in OBJ: {mesh_path}")

        return submeshes, np.vstack(vertices_list).astype(np.float32, copy=False)

    @classmethod
    def _parse_glb_chunks(cls, mesh_path: str) -> tuple[dict, bytes]:
        with open(mesh_path, "rb") as f:
            data = f.read()

        if len(data) < 20:
            raise ValueError(f"Invalid GLB file: {mesh_path}")

        magic, version, length = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF":
            raise ValueError(f"Invalid GLB magic in file: {mesh_path}")
        if version != 2:
            raise ValueError(f"Unsupported GLB version {version}: {mesh_path}")
        if length > len(data):
            raise ValueError(f"Corrupted GLB length header: {mesh_path}")

        offset = 12
        json_chunk = None
        bin_chunk = b""
        while offset + 8 <= len(data):
            chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
            offset += 8
            chunk = data[offset:offset + chunk_length]
            offset += chunk_length
            if chunk_type == b"JSON":
                json_chunk = json.loads(chunk.decode("utf-8"))
            elif chunk_type.startswith(b"BIN"):
                bin_chunk = bytes(chunk)

        if json_chunk is None:
            raise ValueError(f"GLB missing JSON chunk: {mesh_path}")
        return json_chunk, bin_chunk

    @classmethod
    def _load_glb_submeshes(cls, mesh_path: str) -> tuple[list[dict], np.ndarray]:
        gltf, bin_chunk = cls._parse_glb_chunks(mesh_path)
        buffers = [bin_chunk]
        submeshes: List[dict] = []
        vertices_list: List[np.ndarray] = []
        texture_cache: dict[int, int] = {}

        images = gltf.get("images", [])
        textures = gltf.get("textures", [])
        materials = gltf.get("materials", [])
        meshes = gltf.get("meshes", [])
        nodes = gltf.get("nodes", [])
        scenes = gltf.get("scenes", [])
        default_scene_idx = int(gltf.get("scene", 0)) if scenes else -1

        def resolve_texture(texture_idx: int | None) -> int | None:
            if texture_idx is None or texture_idx < 0 or texture_idx >= len(textures):
                return None
            cached = texture_cache.get(texture_idx)
            if cached is not None:
                return cached

            texture = textures[texture_idx]
            image_idx = texture.get("source")
            if image_idx is None or image_idx < 0 or image_idx >= len(images):
                return None

            image = images[image_idx]
            image_bytes: bytes | None = None
            if "bufferView" in image:
                buffer_view = gltf["bufferViews"][image["bufferView"]]
                source_buffer = buffers[buffer_view.get("buffer", 0)]
                start = int(buffer_view.get("byteOffset", 0))
                end = start + int(buffer_view["byteLength"])
                image_bytes = source_buffer[start:end]
            elif "uri" in image:
                image_path = os.path.join(os.path.dirname(mesh_path), image["uri"])
                if os.path.exists(image_path):
                    return cls._create_gl_texture(image_path)

            tex_id = cls._create_gl_texture_from_bytes(image_bytes or b"")
            if tex_id is not None:
                texture_cache[texture_idx] = tex_id
            return tex_id

        scene_nodes = scenes[default_scene_idx]["nodes"] if 0 <= default_scene_idx < len(scenes) else list(range(len(nodes)))

        def traverse(node_idx: int, parent_matrix: np.ndarray):
            node = nodes[node_idx]
            world_matrix = parent_matrix @ cls._gltf_node_matrix(node)

            mesh_idx = node.get("mesh")
            if mesh_idx is not None:
                mesh_def = meshes[mesh_idx]
                for primitive in mesh_def.get("primitives", []):
                    if primitive.get("mode", 4) != 4:
                        continue

                    attrs = primitive.get("attributes", {})
                    position_accessor = attrs.get("POSITION")
                    if position_accessor is None:
                        continue

                    positions = cls._read_gltf_accessor(gltf, position_accessor, buffers)
                    normals = cls._read_gltf_accessor(gltf, attrs["NORMAL"], buffers) if "NORMAL" in attrs else None
                    texcoords = cls._read_gltf_accessor(gltf, attrs["TEXCOORD_0"], buffers) if "TEXCOORD_0" in attrs else None
                    indices = cls._read_gltf_accessor(gltf, primitive["indices"], buffers) if "indices" in primitive else None

                    v, n, t = cls._expand_indexed_primitive(positions, normals, texcoords, indices)
                    v = cls._transform_positions(v, world_matrix)
                    n = cls._transform_normals(n, world_matrix)

                    material_idx = primitive.get("material")
                    material = materials[material_idx] if material_idx is not None and material_idx < len(materials) else {}
                    pbr = material.get("pbrMetallicRoughness", {})
                    base_color_factor = np.array(pbr.get("baseColorFactor", [0.6, 0.6, 0.6, 1.0])[:3], dtype=np.float32)
                    base_color_texture = pbr.get("baseColorTexture", {})
                    tex_id = resolve_texture(base_color_texture.get("index"))
                    emissive_factor = np.array(material.get("emissiveFactor", [0.0, 0.0, 0.0]), dtype=np.float32)
                    emissive_strength = float(np.linalg.norm(emissive_factor))
                    material_name = material.get("name", f"material_{material_idx if material_idx is not None else 0}")

                    colors = np.tile(base_color_factor[None, :], (v.shape[0], 1)).astype(np.float32)

                    submesh_vao = VAO()
                    submesh_vao.add_vbo(0, v, ncomponents=3)
                    submesh_vao.add_vbo(1, n, ncomponents=3)
                    submesh_vao.add_vbo(2, t, ncomponents=2)
                    submesh_vao.add_vbo(3, colors, ncomponents=3)
                    submeshes.append(
                        {
                            "vao": submesh_vao,
                            "vertex_count": int(v.shape[0]),
                            "texture_id": tex_id,
                            "has_texture": tex_id is not None,
                            "base_color": base_color_factor,
                            "material_name": material_name,
                            "emissive_color": emissive_factor,
                            "emissive_strength": emissive_strength,
                            "vertices": v.astype(np.float32, copy=False),
                        }
                    )
                    vertices_list.append(v)

            for child_idx in node.get("children", []):
                traverse(child_idx, world_matrix)

        for node_idx in scene_nodes:
            traverse(node_idx, np.eye(4, dtype=np.float32))

        if not vertices_list:
            raise ValueError(f"No vertices found in GLB: {mesh_path}")

        return submeshes, np.vstack(vertices_list).astype(np.float32, copy=False)

    @staticmethod
    def _filter_submeshes_by_extent(submeshes: list[dict], max_extent: float) -> list[dict]:
        """
        Keep only submeshes whose local AABB is not larger than max_extent on any axis.
        Useful for imported vehicle GLBs that include huge helper/floor geometry.
        """
        threshold = float(max_extent)
        kept: list[dict] = []
        for sm in submeshes:
            verts = sm.get("vertices")
            if verts is None or len(verts) == 0:
                continue
            mins = np.min(verts, axis=0)
            maxs = np.max(verts, axis=0)
            extent = maxs - mins
            if float(np.max(extent)) <= threshold:
                kept.append(sm)
        return kept

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

    def apply_mtl_overrides(self, mtl_path: str):
        """
        Apply MTL material overrides to already loaded submeshes.
        This is useful when loading geometry from formats like FBX/BLEND
        while still wanting to enforce a companion MTL look definition.
        """
        material_overrides = self._parse_mtl_materials(mtl_path)
        if not material_overrides:
            return

        texture_cache: dict[str, int] = {}
        for sm in self.submeshes:
            material_name = sm.get("material_name")
            if not material_name:
                continue

            override = material_overrides.get(material_name)
            if not override:
                continue

            base_color = override.get("diffuse")
            if base_color is not None:
                sm["base_color"] = np.asarray(base_color, dtype=np.float32)

            emissive_color = np.asarray(override.get("emissive", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            sm["emissive_color"] = emissive_color
            sm["emissive_strength"] = float(np.linalg.norm(emissive_color))

            texture_path = override.get("texture_path")
            if texture_path and not sm.get("has_texture", False):
                tex_id = texture_cache.get(texture_path)
                if tex_id is None:
                    tex_id = self._create_gl_texture(texture_path)
                    if tex_id is not None:
                        texture_cache[texture_path] = tex_id
                if tex_id is not None:
                    sm["texture_id"] = tex_id
                    sm["has_texture"] = True

        self.texture_id = next((sm["texture_id"] for sm in self.submeshes if sm.get("texture_id") is not None), None)
        self.has_texture = any(bool(sm.get("has_texture", False)) for sm in self.submeshes)

    def setup(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_path = self.filename
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(base_dir, mesh_path)

        ext = os.path.splitext(mesh_path)[1].lower()
        if ext == ".obj":
            self.submeshes, self.vertices = self._load_obj_submeshes(mesh_path)
        elif ext == ".glb":
            self.submeshes, self.vertices = self._load_glb_submeshes(mesh_path)
        else:
            raise ValueError(f"Unsupported mesh format: {ext}")

        if self.max_submesh_extent is not None:
            filtered = self._filter_submeshes_by_extent(self.submeshes, self.max_submesh_extent)
            if filtered:
                self.submeshes = filtered
                self.vertices = np.vstack([sm["vertices"] for sm in self.submeshes]).astype(np.float32, copy=False)

        vertices = self.vertices
        self.vertex_count = int(vertices.shape[0])
        self.local_aabb_min, self.local_aabb_max = self.compute_local_aabb()

        # Backward-compatible summary flags for callers that only need to know
        # whether any submesh contains a texture.
        self.texture_id = next((sm["texture_id"] for sm in self.submeshes if sm["texture_id"] is not None), None)
        self.has_texture = any(sm["has_texture"] for sm in self.submeshes)

        return self

    def draw(self, projection: np.ndarray, view: np.ndarray, model: np.ndarray | None, shader):
        """Generic draw using an external shader (RGB/Mask/Depth passes)."""
        if model is None:
            model = identity()

        modelview = view @ model

        uma = UManager(shader)
        uma.upload_uniform_matrix4fv(projection, "projection", transpose=True)
        uma.upload_uniform_matrix4fv(modelview, "modelview", transpose=True)

        # Optional for shaders that may use world model matrix.
        uma.upload_uniform_matrix4fv(model, "model", transpose=True)

        for submesh in self.submeshes:
            texture_id = submesh["texture_id"]
            if texture_id is not None:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
                uma.upload_uniform_scalar1i(0, "diffuse_map")
                uma.upload_uniform_scalar1i(1, "has_texture")
            else:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                uma.upload_uniform_scalar1i(0, "has_texture")

            uma.upload_uniform_scalar1i(1, "use_uniform_base_color")
            uma.upload_uniform_vector3fv(submesh["base_color"], "base_color")
            emissive_color = np.asarray(submesh.get("emissive_color", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            emissive_strength = float(submesh.get("emissive_strength", 0.0))
            uma.upload_uniform_scalar1i(1 if emissive_strength > 1e-5 else 0, "use_emissive")
            uma.upload_uniform_vector3fv(emissive_color, "emissive_color")
            uma.upload_uniform_scalar1f(emissive_strength, "emissive_strength")

            submesh["vao"].activate()
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, submesh["vertex_count"])
            submesh["vao"].deactivate()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
