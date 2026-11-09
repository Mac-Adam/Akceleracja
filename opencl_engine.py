import pyopencl as cl
import numpy as np
from pathlib import Path


class OpenCLEngine:
    def __init__(self, kernel_path: str):
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError(
                "No OpenCL platforms found. Check your ICD loader/drivers."
            )

        devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
        if not devices:
            devices = platforms[0].get_devices(device_type=cl.device_type.ALL)

        self.device = devices[0]
        print(f"[OpenCL] Using Device: {self.device.name} ({self.device.vendor})")

        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        # Load and Compile Kernel
        kernel_source = Path(kernel_path).read_text(encoding="utf-8")
        self.program = cl.Program(self.ctx, kernel_source).build()
        self.median_kernel = cl.Kernel(self.program, "median_kernel")

        # Image placeholders (lazily initialized on first frame)
        self.d_input_img = None
        self.d_output_img = None
        self.frame_shape = None
        self.host_output = None

    def _allocate_images(self, shape: tuple, dtype: np.dtype):
        """Pre-allocates 2D OpenCL Image objects (RGBA, uint8) matching median_kernel.cl."""
        self.frame_shape = shape
        height, width = shape[:2]

        # CL_RGBA + CL_UNSIGNED_INT8
        img_format = cl.ImageFormat(
            cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8
        )
        mf = cl.mem_flags

        # Note: OpenCL Image shape is specified as (width, height)
        self.d_input_img = cl.Image(
            self.ctx, mf.READ_ONLY, img_format, shape=(width, height)
        )
        self.d_output_img = cl.Image(
            self.ctx, mf.WRITE_ONLY, img_format, shape=(width, height)
        )
        self.host_output = np.empty((height, width, 4), dtype=dtype)

        # Set the 2 image2d_t arguments
        self.median_kernel.set_arg(0, self.d_input_img)
        self.median_kernel.set_arg(1, self.d_output_img)

    def apply_median_filter(self, bgra_frame: np.ndarray) -> np.ndarray:
        """
        Transfers a 4-channel (H, W, 4) uint8 frame to an OpenCL image2d_t,
        executes median_kernel, and copies the 4-channel result back to host memory.
        """
        if not bgra_frame.flags["C_CONTIGUOUS"]:
            bgra_frame = np.ascontiguousarray(bgra_frame)

        if self.frame_shape != bgra_frame.shape:
            self._allocate_images(bgra_frame.shape, bgra_frame.dtype)

        height, width = bgra_frame.shape[:2]
        origin = (0, 0, 0)
        region = (width, height, 1)

        # Host -> Device Image transfer
        cl.enqueue_copy(
            self.queue,
            self.d_input_img,
            bgra_frame,
            origin=origin,
            region=region,
            is_blocking=False,
        )

        # Execute Kernel: Global work size is (width, height)
        cl.enqueue_nd_range_kernel(
            self.queue, self.median_kernel, (width, height), None
        )

        # Device -> Host Image transfer
        cl.enqueue_copy(
            self.queue,
            self.host_output,
            self.d_output_img,
            origin=origin,
            region=region,
            is_blocking=True,
        )

        return self.host_output
