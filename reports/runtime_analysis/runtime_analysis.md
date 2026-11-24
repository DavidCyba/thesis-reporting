python speed throtteling when windows not in focus, administator does not help
fixed drivers
radeon relive off
task manager worst gpu offenders off


Do multiple runs and take mean / std

Benchmark:  DW - Using HCRes_VarAdapt 4 layers with 7000 steps and pathtype 2, Mass = 2.0
            92.55s ~ 75.63 it / s 
            Measurement starting before first iteration and ending immediatly after last (maybe in future start after a warmup)



| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| RAM | 64 GB |
| GPU | AMD Radeon RX 9070 XT, 16 GB VRAM |
| Operating system | Windows 11, build 26200, x64 |
| GPU driver | AMD Adrenalin 26.11.1 |
| Python | 3.12.10 |
| PyTorch | `2.9.1+rocm7.2.1` |
| HIP runtime | `7.2.53211-158bd99533` |
| PyTorch CPU intra-operation threads | 8 |