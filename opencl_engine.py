import pyopencl as cl
import numpy as np
from pathlib import Path


class OpenCLEngine:
    def __init__(
        self,
        kernel_path: str = "gmm_kernel.cl",
        num_gaussians: int = 3,
        alpha: float = 0.01,
        bg_thresh: float = 0.7,
        match_thresh: float = 2.5,
        init_var: float = 225.0,
        min_var: float = 9.0,
    ):
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be >= 1.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if not 0.0 < bg_thresh < 1.0:
            raise ValueError("bg_thresh must be in (0, 1).")
        if match_thresh <= 0.0:
            raise ValueError("match_thresh must be positive.")
        if init_var <= 0.0 or min_var <= 0.0:
            raise ValueError("init_var and min_var must be positive.")

        self.num_gaussians = int(num_gaussians)
        self.alpha = float(alpha)
        self.bg_thresh = float(bg_thresh)
        self.match_thresh = float(match_thresh)
        self.init_var = float(init_var)
        self.min_var = float(min_var)

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

        # Load and Compile GMM Kernel
        kernel_source = Path(kernel_path).read_text(encoding="utf-8")
        self.program = cl.Program(self.ctx, kernel_source).build()
        self.gmm_kernel = cl.Kernel(self.program, "gmm_kernel")

        # Input image placeholder (lazily initialized on first frame)
        self.d_input_img = None
        self.frame_shape = None

        # GMM state placeholders (lazily initialized on first frame)
        self.gmm_means_buf = None
        self.gmm_vars_buf = None
        self.gmm_weights_buf = None
        self.gmm_mask_buf = None
        self.gmm_shape = None
        self.host_mask = None
        self.gmm_first_frame = True

    def _allocate_images(self, shape: tuple):
        """Pre-allocates the 2D OpenCL input Image (RGBA, uint8)."""
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

    def _allocate_gmm(self, height: int, width: int):
        """Pre-allocates float32 GMM state (means/vars/weights) and the mask."""
        mf = cl.mem_flags
        n = height * width * self.num_gaussians
        zeros_f = np.zeros(n, dtype=np.float32)
        zeros_b = np.zeros(height * width, dtype=np.uint8)
        self.gmm_means_buf = cl.Buffer(
            self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros_f
        )
        self.gmm_vars_buf = cl.Buffer(
            self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros_f
        )
        self.gmm_weights_buf = cl.Buffer(
            self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros_f
        )
        self.gmm_mask_buf = cl.Buffer(
            self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros_b
        )
        self.host_mask = np.empty((height, width), dtype=np.uint8)
        self.gmm_shape = (height, width)
        self.gmm_first_frame = True

    def reset_gmm(self):
        """Force GMM re-initialization from the next frame."""
        self.gmm_first_frame = True

    def apply_gmm(self, bgra_frame: np.ndarray) -> np.ndarray:
        """
        Runs one GMM background-subtraction step on a 4-channel (H, W, 4)
        uint8 frame and returns the (H, W) uint8 foreground mask
        (0 = background, 255 = foreground).
        """
        if not bgra_frame.flags["C_CONTIGUOUS"]:
            bgra_frame = np.ascontiguousarray(bgra_frame)

        if self.frame_shape != bgra_frame.shape:
            self._allocate_images(bgra_frame.shape)

        height, width = bgra_frame.shape[:2]
        if self.gmm_shape != (height, width):
            self._allocate_gmm(height, width)

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

        self.gmm_kernel.set_arg(0, self.d_input_img)
        self.gmm_kernel.set_arg(1, self.gmm_means_buf)
        self.gmm_kernel.set_arg(2, self.gmm_vars_buf)
        self.gmm_kernel.set_arg(3, self.gmm_weights_buf)
        self.gmm_kernel.set_arg(4, self.gmm_mask_buf)
        self.gmm_kernel.set_arg(5, np.int32(self.num_gaussians))
        self.gmm_kernel.set_arg(6, np.float32(self.alpha))
        self.gmm_kernel.set_arg(7, np.float32(self.bg_thresh))
        self.gmm_kernel.set_arg(8, np.float32(self.match_thresh))
        self.gmm_kernel.set_arg(9, np.float32(self.init_var))
        self.gmm_kernel.set_arg(10, np.float32(self.min_var))
        self.gmm_kernel.set_arg(11, np.int32(1 if self.gmm_first_frame else 0))

        # Execute Kernel: Global work size is (width, height)
        cl.enqueue_nd_range_kernel(self.queue, self.gmm_kernel, (width, height), None)

        # Device -> Host mask transfer
        cl.enqueue_copy(
            self.queue,
            self.host_mask,
            self.gmm_mask_buf,
            is_blocking=True,
        )
        self.gmm_first_frame = False

        return self.host_mask
