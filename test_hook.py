import glfw
from viewer import ViewerApp
def run_test():
    app = ViewerApp(width=800, height=600)
    app._run_auto_generate_hook(num_frames=2)
    glfw.terminate()
run_test()
