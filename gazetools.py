import pyopencl as cl
import numpy as np

def loadProgram(filename):
    with open(filename, 'r') as f:
        return "".join(f.readlines())

class gazetools:

    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.cl_distance_2_point = cl.Program(self.ctx, loadProgram("distance_2_point.cl")).build()
        self.cl_subtended_angle = cl.Program(self.ctx, loadProgram("subtended_angle.cl")).build()

    def distance_2_point(self, x, y, rx, ry, sw, sh, ez, ex, ey):
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        ez = np.array(ez, dtype=np.float32)
        ex = np.array(ex, dtype=np.float32)
        ey = np.array(ey, dtype=np.float32)
        rx = np.float32(rx)
        ry = np.float32(ry)
        sw = np.float32(sw)
        sh = np.float32(sh)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        y_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y)
        ez_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ez)
        ex_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ex)
        ey_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ey)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, x.nbytes)
        self.cl_distance_2_point.distance_2_point(self.queue, (512,), None, x_buf, y_buf, rx, ry, sw, sh, ez_buf, ex_buf, ey_buf, out_buf)
        out = np.empty_like(x)
        cl.enqueue_read_buffer(self.queue, out_buf, out).wait()
        return out

    def subtended_angle(self, x1, y1, x2, y2, rx, ry, sw, sh, ez, ex, ey):
        x1 = np.array(x1, dtype=np.float32)
        y1 = np.array(y1, dtype=np.float32)
        x2 = np.array(x2, dtype=np.float32)
        y2 = np.array(y2, dtype=np.float32)
        ez = np.array(ez, dtype=np.float32)
        ex = np.array(ex, dtype=np.float32)
        ey = np.array(ey, dtype=np.float32)
        rx = np.float32(rx)
        ry = np.float32(ry)
        sw = np.float32(sw)
        sh = np.float32(sh)
        x1_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x1)
        y1_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y1)
        x2_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x2)
        y2_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y2)
        ez_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ez)
        ex_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ex)
        ey_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ey)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, x1.nbytes)
        self.cl_subtended_angle.subtended_angle(self.queue, (512,), None, x1_buf, y1_buf, x2_buf, y2_buf, rx, ry, sw, sh, ez_buf, ex_buf, ey_buf, out_buf)
        out = np.empty_like(x1)
        cl.enqueue_read_buffer(self.queue, out_buf, out).wait()
        return out

if __name__ == "__main__":
    GT = gazetools()
    print GT.distance_2_point((0.0,0.0,1680.0/2.0,1680.0/2.0),
                              (0.0,1050.0/2.0,0.0,1050/2.0),
                              1680.0,1050.0,473.76,296.1,
                              [700.0]*4,[0.0]*4,[0.0]*4)
    print GT.distance_2_point((1680/2),
                              (1050/2),
                              1680.0,1050.0,473.76,296.1,
                              [700]*1,[0]*1,[0]*1)
    print GT.subtended_angle((0),
                             (1050/2),
                             (1680),
                             (1050/2),
                             1680,1050,473.76,296.1,
                             [700]*1,[0]*1,[0]*1)
