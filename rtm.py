import drjit as dr
from drjit.auto import TensorXf, Quaternion4f, Array3f, Float
import mitsuba as mi
from imgui_bundle import imgui, immapp
# dr.set_flag(dr.JitFlag.Debug, True)
mi.set_variant('cuda_ad_rgb' if dr.has_backend(dr.JitBackend.CUDA) else 'llvm_ad_rgb')

class GlTexture:
    def __init__(self, width, height):
        import OpenGL.GL as gl
        self.width, self.height, self.id = width, height, gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0,
            gl.GL_RGBA, gl.GL_FLOAT, None
        )
        try:
            self.interop = dr.cuda.GLInterop.from_texture(self.id)
        except Exception as e:
            print("Warning: Could not create CUDA-OpenGL interop:", e)
            self.interop = None

    def upload(self, img):
        if dr.backend_v(img) == dr.JitBackend.CUDA:
            self.interop.map().upload(img).unmap()
        elif dr.backend_v(img) == dr.JitBackend.LLVM:
            import OpenGL.GL as gl
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                gl.GL_RGBA, gl.GL_FLOAT, img.to_numpy()
            )

class Camera:
    def __init__(self):
        self.q, self.c, self.d = Quaternion4f(0, 0, 0, 1), Array3f(0, 0, 0), Float(14.0)
        self.resolution, self.scale = dr.scalar.Array2u(256), 1.0
        self.gl_texture: GlTexture | None = None
        self.start_pos = None
        self.sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': mi.ScalarTransform4f(),
            'sampler': {
                'type': 'independent',
                'sample_count': 1
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.resolution[0],
                'height': self.resolution[1],
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })
        self.params = mi.traverse(self.sensor)
        self.update()

    def map_to_sphere(self, px, py):
        x = -(1.0 - (px / (self.resolution[0] * self.scale)) * 2.0)
        y = (1.0 - (py / (self.resolution[1] * self.scale)) * 2.0)
        length2 = x*x + y*y
        if length2 > 1.0:
            return Array3f(x, y, 0.0) / dr.sqrt(length2)
        else:
            return Array3f(x, y, dr.sqrt(max(0.0, 1.0 - length2)))

    def rotate(self, a, b):
        a = self.map_to_sphere(*a)
        b = self.map_to_sphere(*b)
        perp = dr.cross(b, a)
        if dr.norm(perp) > 1e-5:
            q = Quaternion4f(perp.x, perp.y, perp.z, dr.dot(a, b))
        else:
            q = Quaternion4f(0, 0, 0, 1)
        return dr.normalize(self.q * q)

    def update(self, q=Quaternion4f(0, 0, 0, 1)):
        self.to_world = mi.Transform4f().look_at(
                origin=dr.quat_apply(q, Array3f(0, 0, self.d)) + self.c,
                target=self.c,
                up=dr.quat_apply(q, Array3f(0, 1, 0))
            )
        self.params['to_world'] = self.to_world
        self.params['film.size'] = self.resolution
        self.film = dr.zeros(TensorXf, shape=(self.resolution[1], self.resolution[0], 4))
        self.params.update()

    def handle_gl_texture(self):
        ' Only call if gl texture is needed and with a valid opengl context '
        if not self.gl_texture or self.gl_texture.width != self.resolution[0] or self.gl_texture.height != self.resolution[1]:
            self.gl_texture = GlTexture(self.resolution[0], self.resolution[1])

    def process_imgui_inputs(self, min, max):
        from imgui_bundle import imgui
        io = imgui.get_io()
        if imgui.is_mouse_clicked(0):
            self.start_pos = io.mouse_pos - min
        if imgui.is_mouse_released(0) and self.start_pos:
            self.q = self.rotate(self.start_pos, io.mouse_pos - min)
            self.update(self.q)
            self.start_pos = None
        if imgui.is_mouse_dragging(0) and self.start_pos:
            self.update(self.rotate(self.start_pos, io.mouse_pos - min))
        elif imgui.is_mouse_dragging(1):
            self.c += dr.quat_apply(self.q, Array3f(-io.mouse_delta.x, io.mouse_delta.y, 0.0)) * (self.d * 0.002)
            self.update(self.q)
        if io.mouse_wheel != 0:
            self.d = self.d - io.mouse_wheel * 0.3
            if self.d < 0.1:
                self.d = 0.1
            self.update(self.q)
        
class State:
    def __init__(self, scene_file):
        self.scene = mi.load_file(scene_file)
        self.frame = dr.opaque(dr.auto.UInt, 0)
        self.accumulate = True
        self.camera = Camera()

    def process_inputs(self, min, max):
        if imgui.table_get_hovered_column() == 0:
            self.camera.process_imgui_inputs(min, max)

@dr.freeze(warn_after=1)
def render(scene, texture, seed=0, sensor=0):
    return mi.render(scene, spp=1, seed=seed, sensor=sensor)

def show_image(width, height):
    state.camera.handle_gl_texture()
    img = render(state.scene, state.camera.gl_texture, seed=state.frame, sensor=state.camera.sensor)
    if img.shape[2] == 3:
        # Add alpha channel if not present
        alpha = dr.ones(TensorXf, shape=(img.shape[0], img.shape[1], 1))
        img = dr.concat((TensorXf(img), alpha), axis=2)
    state.camera.film = state.camera.film + img if state.accumulate else img
    # use alpha channel to store the sample count
    state.camera.gl_texture.upload(dr.linear_to_srgb(state.camera.film / state.camera.film[0][0][3]) )
    state.frame += 1
    w, h = width * state.camera.scale, height * state.camera.scale
    imgui.set_cursor_pos_x(imgui.get_column_width()//2 - w//2)
    imgui.set_cursor_pos_y(imgui.get_window_viewport().size.y//2 - h//2)
    imgui.image(imgui.ImTextureRef(state.camera.gl_texture.id), (w, h))
    state.process_inputs((imgui.get_column_width()//2 - w//2, imgui.get_window_viewport().size.y//2 - h//2), (imgui.get_column_width()//2 + w//2, imgui.get_window_viewport().size.y//2 + h//2))

def gui():
    imgui.set_next_window_size(imgui.get_io().display_size)
    imgui.set_next_window_pos((0, 0))
    if imgui.begin("##FullscreenWindow", None, imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_focus_on_appearing):
        if imgui.begin_table("Table", 2, imgui.TableFlags_.resizable | imgui.TableFlags_.sizing_stretch_prop, (-1, -1)):
            imgui.table_setup_column("Rendering", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("Settings", imgui.TableColumnFlags_.width_fixed, 400)
            imgui.table_next_row()
            imgui.table_set_column_index(0)
            show_image(*state.camera.resolution)
            imgui.table_set_column_index(1)
            if imgui.begin_tab_bar("Settings"):
                if imgui.begin_tab_item("General")[0]:
                    if imgui.collapsing_header("Film", imgui.TreeNodeFlags_.default_open):
                        _, state.camera.scale = imgui.slider_float("Scale", state.camera.scale, 1.0, 10.0)
                        resolutionChanged, state.camera.resolution[0] = imgui.slider_int("Resolution", state.camera.resolution[0], 64, 2048)
                        if resolutionChanged:
                            state.camera.resolution[1] = state.camera.resolution[0]
                            state.camera.update(state.camera.q)
                    state.accumulate = imgui.checkbox("Accumulate", state.accumulate)[1]
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Other")[0]:
                    imgui.text("More settings can be added here.")
                    imgui.end_tab_item()
                imgui.end_tab_bar()
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + imgui.get_content_region_avail().y - imgui.get_text_line_height_with_spacing())
            imgui.separator()
            imgui.text(f"{imgui.get_io().framerate :.2f} FPS | { 1000.0 / imgui.get_io().framerate :.1f} ms")
            imgui.end_table()
        imgui.end()

if __name__ == "__main__":
    state = State("scenes/simple.xml")
    immapp.run(gui_function=gui, window_size=(800, 400), window_title="Renderer", fps_idle=0)