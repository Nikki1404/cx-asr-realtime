(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime# nvidia-smi
Tue Feb  3 09:29:02 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    Off |   00000000:00:1E.0 Off |                    0 |
|  0%   27C    P0             56W /  300W |   22548MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A    983594      C   tritonserver                                 9548MiB |
|    0   N/A  N/A   2417379      C   /usr/local/bin/python3.12                    4508MiB |
|    0   N/A  N/A   2825985      C   python3                                      1922MiB |
|    0   N/A  N/A   3600453      C   /usr/bin/python3                             2814MiB |
|    0   N/A  N/A   3901281      C   /usr/bin/python3                             3726MiB |
+-----------------------------------------------------------------------------------------+
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime#
