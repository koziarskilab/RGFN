# SciNet support report — balam008 NVIDIA OpenCL broken

Draft email to `balam-support@scinet.utoronto.ca`. Minimal reproducer:
`opencl_healthcheck.c` (this directory).

---

**Subject:** balam008 — NVIDIA OpenCL broken (clCreateContext fails), CUDA fine

Hi,

I think **balam008** has a wedged NVIDIA OpenCL runtime. On that node, creating
any OpenCL context fails with `CL_OUT_OF_RESOURCES (-5)`, even though CUDA works
normally (my PyTorch/gnina jobs run fine there). The same code works on other
Balam compute nodes, so it looks node-specific rather than a driver/image problem.

**Evidence** — a ~20-line OpenCL program that just does
`clGetPlatformIDs → clGetDeviceIDs(GPU) → clCreateContext`, run via
`srun --gpus-per-node=1`:

- **balam008** (A100-SXM4): platform + device found, then `clCreateContext err=-5 FAIL`
- **balam009** (A100-SXM4, same driver 580 / CUDA 13): `clCreateContext err=0 SUCCESS`
- login node (A100-PCIE): SUCCESS

It fails on balam008 regardless of `CUDA_VISIBLE_DEVICES` (0 / UUID / unset) and
regardless of which OpenCL loader I use (system `/lib64`, cuda/11.8.0, or
cuda/12.3.1 module). No MPS running, compute mode is Default, MIG disabled.

**Impact:** OpenCL applications (in my case QuickVina2-GPU for molecular docking)
**fail silently** on balam008 — they produce no output rather than an obvious
crash, so a job can waste its whole allocation before you notice.

**Ask:** could you reset / power-cycle the GPU + OpenCL state on balam008 (e.g.
reload the NVIDIA kernel modules / restart the node)? Happy to share the minimal
reproducer C file if useful.

Thanks,
Mark Stevens
