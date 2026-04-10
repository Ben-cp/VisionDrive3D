# VisionDrive3D

# BTL Computer Graphics — Bộ sinh ảnh tổng hợp cho AI

Dự án môn **Đồ họa máy tính (Computer Graphics)**: xây dựng pipeline **render cảnh 3D** để tạo **ảnh tổng hợp** (synthetic images) phục vụ huấn luyện/đánh giá AI.

> Thư mục làm việc chính: `BTL2/`

---

## Mục tiêu

- Render các cảnh 3D có ánh sáng, camera, vật liệu cơ bản.
- Hỗ trợ nạp mesh từ file (Giai đoạn 1: `.obj`).
- Mở rộng dần để sinh dataset (nhiều góc nhìn, nhiều điều kiện ánh sáng, xuất ảnh + annotation nếu cần).

---

## Cấu trúc thư mục

```
BTL2/
  libs/
    buffer.py      # VAO, UManager (upload uniforms, textures)
    shader.py      # Shader (compile/link GLSL)
    transform.py   # Trackball + các hàm ma trận (identity, perspective, ...)
    camera.py      # Camera (extends Trackball)
    lighting.py    # LightingManager, Light, Material
  phong.vert       # Phong shader (vertex)
  phong.frag       # Phong shader (fragment)
  mesh.py          # Mesh loader OBJ (pywavefront) + VAO setup
  viewer.py        # GLFW viewer tối giản để chạy demo
  car.obj          # (KHÔNG có sẵn) bạn tự tải/đặt vào đây để test
```

---

## Quy ước kiến trúc (Design Pattern)

Mọi đối tượng vẽ được (**drawable**) cần có:

- **`setup(self)`**: khởi tạo OpenGL resources (shader, VAO/VBO/EBO, texture...).  
  Hàm này **trả về `self`** để hỗ trợ method chaining.
- **`draw(self, projection, view, model)`**: render đối tượng.

Trong vòng lặp render, `viewer` sẽ gọi:

```python
drawable.draw(projection, view, None)
```

Vì vậy các drawable nên xử lý `model=None` (mặc định `identity()`).

---

## Shader Phong (đang dùng)

File: `BTL2/phong.vert`, `BTL2/phong.frag`

### Vertex attributes (bắt buộc)

- location **0**: `position` (vec3)
- location **1**: `normal` (vec3)
- location **2**: `texcoord` (vec2)
- location **3**: `color` (vec3)

### Uniforms (LightingManager đang upload)

`LightingManager(uma).setup_phong()` sẽ upload các uniform:

- `projection` (mat4)
- `modelview` (mat4)
- `I_light` (mat3)
- `light_pos` (vec3)
- `K_materials` (mat3)
- `shininess` (float)
- `mode` (int)

---

## Cài đặt môi trường

### Yêu cầu hệ thống

- Python **3.9+** (khuyến nghị 3.10+)
- GPU/driver hỗ trợ **OpenGL 3.3+**

### Cài package (Windows / PowerShell)

Từ thư mục workspace:

```bash
pip install numpy glfw PyOpenGL PyOpenGL_accelerate opencv-python pywavefront
```

Ghi chú:
- `opencv-python` đang cần vì `BTL2/libs/buffer.py` import `cv2` (dùng cho texture loader).

---

## Chạy demo Mesh Loader (Giai đoạn 1)

### 1) Chuẩn bị OBJ để test

Đặt file `car.obj` vào **`BTL2/car.obj`**.

### 2) Chạy viewer

```bash
cd BTL2
python viewer.py
```

### Controls (Viewer cơ bản)

- **Mouse Left drag**: xoay (trackball)
- **Mouse Right drag**: pan
- **Mouse Wheel**: zoom
- **W**: toggle `wireframe/point/fill`
- **Q** hoặc **Esc**: thoát

---

## Giai đoạn 1 — Mesh Loader (`OBJ`) (đã tích hợp)

File chính:

- `BTL2/mesh.py`: class `Mesh(filename)`
  - `setup()` đọc OBJ bằng `pywavefront`, tách mảng interleaved của `material.vertices`, tạo các buffer:
    - `vertices (N,3)` → location 0
    - `normals (N,3)` → location 1
    - `texcoords (N,2)` → location 2 (nếu thiếu → zeros)
    - `colors (N,3)` → location 3 (mặc định xám `[0.6,0.6,0.6]`)
  - `draw()` upload `projection`, `modelview`, setup phong lighting, rồi `glDrawArrays(GL_TRIANGLES, ...)`.

---

## Thêm một drawable mới (workflow chuẩn)

1) Tạo file `BTL2/<object>.py` và class mới.
2) Trong `setup()`:
   - compile shader (hoặc dùng shader có sẵn)
   - tạo `VAO()` và `add_vbo()` đúng các location attribute mà shader dùng
3) Trong `draw()`:
   - upload `projection`, `modelview` (và uniform khác nếu cần)
   - bind VAO → draw → unbind
4) Trong `viewer.py`:
   - `obj = YourObject(...).setup()`
   - `viewer.add(obj)`

---

## Troubleshooting nhanh

- **`ModuleNotFoundError: No module named 'OpenGL'`**  
  Cài lại: `pip install PyOpenGL PyOpenGL_accelerate`

- **Cửa sổ mở nhưng màn hình xám/đen**  
  Kiểm tra:
  - shader compile/link có lỗi (log sẽ in ra console)
  - OBJ có dữ liệu vertex/normal không
  - camera đang nhìn thấy object (thử zoom out)

- **Thiếu file OBJ**  
  `viewer.py` đang load `Mesh("car.obj")` theo đường dẫn tương đối trong `BTL2/`.

---

## Quy ước làm việc nhóm

- **Không commit** các asset nặng nếu không cần (OBJ/texture lớn) — ưu tiên để link nguồn tải trong README hoặc `.gitignore` (nếu sau này dùng git).
- Khi thêm shader mới, ghi rõ:
  - attribute locations yêu cầu
  - uniform names cần upload
- Mọi module mới nên bám theo pattern `setup/draw` để `viewer` có thể dùng trực tiếp.
