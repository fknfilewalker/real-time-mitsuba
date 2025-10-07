import drjit as dr
from drjit.cuda.ad import Float, TensorXf, Quaternion4f, Array2u, Array3f
import mitsuba as mi
from imgui_bundle import imgui, immapp
# dr.set_flag(dr.JitFlag.Debug, True)
mi.set_variant('cuda_ad_rgb')

class Texture:
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
        self.interop = dr.cuda.GLInterop.from_texture(self.id)

class Camera:
    def __init__(self):
        self.q, self.c, self.d = dr.scalar.Quaternion4f(0, 0, 0, 1), dr.scalar.Array3f(0, 0, 0), 14.0
        self.resolution, self.scale = dr.scalar.Array2u(256), 1.0
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
            return dr.scalar.Array3f(x, y, 0.0) / dr.sqrt(length2)
        else:
            return dr.scalar.Array3f(x, y, dr.sqrt(max(0.0, 1.0 - length2)))

    def rotate(self, a, b):
        a = self.map_to_sphere(*a)
        b = self.map_to_sphere(*b)
        perp = dr.cross(b, a)
        if dr.norm(perp) > 1e-5:
            q = dr.scalar.Quaternion4f(perp.x, perp.y, perp.z, dr.dot(a, b))
        else:
            q = dr.scalar.Quaternion4f(0, 0, 0, 1)
        return dr.normalize(self.q * q)

    def update(self, q=dr.scalar.Quaternion4f(0, 0, 0, 1)):
        self.to_world = mi.ScalarTransform4f().look_at(
                origin=dr.quat_apply(q, dr.scalar.Array3f(0, 0, self.d)) + self.c,
                target=self.c,
                up=dr.quat_apply(q, dr.scalar.Array3f(0, 1, 0))
            )
        self.params['to_world'] = self.to_world
        self.params['film.size'] = self.resolution
        self.film = dr.zeros(TensorXf, shape=(self.resolution[1], self.resolution[0], 4))
        self.params.update()
        
class State:
    def __init__(self, scene_file):
        self.scene = mi.load_file(scene_file)
        self.frame = 0
        self.accumulate = True
        self.texture: Texture | None = None
        self.camera = Camera()
        self.start_pos = None

    def process_inputs(self, min, max):
        if imgui.table_get_hovered_column() == 0:
            io = imgui.get_io()
            if imgui.is_mouse_clicked(0):
                self.start_pos = io.mouse_pos - min
            if imgui.is_mouse_released(0) and self.start_pos:
                q = self.camera.rotate(self.start_pos, io.mouse_pos - min)
                self.start_pos = None
                self.camera.q = q
                self.camera.update(q)
            if imgui.is_mouse_dragging(0) and self.start_pos:
                self.camera.update(self.camera.rotate(self.start_pos, io.mouse_pos - min))
            elif imgui.is_mouse_dragging(1):
                self.camera.c += dr.quat_apply(self.camera.q, dr.scalar.Array3f(-io.mouse_delta.x, io.mouse_delta.y, 0.0)) * (self.camera.d * 0.002)
                self.camera.update(self.camera.q)
            if io.mouse_wheel != 0:
                self.camera.d = self.camera.d - io.mouse_wheel * 0.3
                if self.camera.d < 0.1:
                    self.camera.d = 0.1
                self.camera.update(self.camera.q)

state = State("scenes/simple.xml")

@dr.freeze
def render(scene, texture, camera, seed=0, sensor=0):
    return mi.render(scene, spp=1, seed=seed, sensor=sensor)

def show_image(width, height):
    if state.texture is None or state.texture.width != width or state.texture.height != height:
        state.texture = Texture(width, height)
    img = render(state.scene, state.texture, state.camera, seed=mi.UInt(state.frame), sensor=state.camera.sensor)
    if img.shape[2] == 3:
        # Add alpha channel if not present
        alpha = dr.ones(TensorXf, shape=(img.shape[0], img.shape[1], 1))
        img = dr.concat((TensorXf(img), alpha), axis=2)
    state.camera.film = state.camera.film + img if state.accumulate else img
    # use alpha channel to store the sample count
    state.texture.interop.map().upload(dr.linear_to_srgb(state.camera.film / state.camera.film[0][0][3]) ).unmap()
    state.frame += 1
    w, h = width * state.camera.scale, height * state.camera.scale
    imgui.set_cursor_pos_x(imgui.get_column_width()//2 - w//2)
    imgui.set_cursor_pos_y(imgui.get_window_viewport().size.y//2 - h//2)
    imgui.image(imgui.ImTextureRef(state.texture.id), (w, h))
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

immapp.run(gui_function=gui, window_size=(800, 400), window_title="Renderer", fps_idle=0)