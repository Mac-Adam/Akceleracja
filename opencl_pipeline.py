from pathlib import Path

import numpy as np
import pyopencl as cl


class Pipeline:
    """BGR uint8 frame in, processed image out. Owns the full OpenCL path."""

    def __init__(self):
        device = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)[0]
        print(f"[OpenCL] Using Device: {device.name}")
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        source = Path(__file__).with_name("median_kernel.cl").read_text(encoding="utf-8")
        self.program = cl.Program(self.ctx, source).build()
        self.median = self.program.median_kernel
        self.frame_shape = None

    def _allocate_images(self, height, width):
        self.frame_shape = (height, width)
        self.input_host = np.empty((height, width, 4), dtype=np.uint8)
        self.input_host[:, :, 3] = 255
        self.output_host = np.empty_like(self.input_host)

        image_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        # Both images can be intermediate inputs/outputs for later kernels.
        self.input_image = cl.Image(self.ctx, cl.mem_flags.READ_WRITE, image_format, shape=(width, height))
        self.output_image = cl.Image(self.ctx, cl.mem_flags.READ_WRITE, image_format, shape=(width, height))

    def process(self, frame):
        """Run the kernel chain; the returned array is reused on the next call."""
        height, width = frame.shape[:2]
        if self.frame_shape != (height, width):
            self._allocate_images(height, width)
        self.input_host[:, :, :3] = frame

        origin = (0, 0, 0)
        region = (width, height, 1)
        cl.enqueue_copy(self.queue, self.input_image, self.input_host, origin=origin, region=region, is_blocking=False)

        self.median(self.queue, (width, height), None, self.input_image, self.output_image)

        cl.enqueue_copy(self.queue, self.output_host, self.output_image, origin=origin, region=region, is_blocking=True)
        return self.output_host
