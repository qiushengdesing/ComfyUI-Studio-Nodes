# ComfyUI-ERP-SeamTools & R2P Controls Suite

360° ERP Panorama Seam Healing, Shaniu 6.0 Algorithm, Polar Structure Guard, & Still Image Upscale Controls for ComfyUI.
专为室内效果图【全景洗图 / 全景放大 / 单张洗图 / 单张放大】研发的 ComfyUI 专属节点套件。

## 核心特性 (Features)
1. **山牛 6.0 原生接缝修复算法 (Shaniu6NativeSeamHeal)**：
   - 纯原生 Python / NumPy / OpenCV 极速高精算法，100% 消除 360° 全景图经纬展开（ERP）左右两端接缝线与色差。
2. **两极与吊顶几何硬装保护 (ERPPoleSourceGuardv3, ERPFullCeilingSourceGuard)**：
   - 保护全景顶棚吊顶、筒灯、地砖硬装在 AI 重绘和超分辨率时不拉伸、不变形、不产生假性弧度。
3. **ERP 画布循环填充与无损还原 (ERPWrapPadv2, ERPWrapRestorev2, ERPOutputResolutionv3)**：
   - 支撑 SeedVR2 等超分辨率放大器无缝处理全景球形图像。
4. **单张图极清放大长短边控制器 (R2PStillUpscaleControls)**：
   - 自动按原图比例计算超清放大长边/短边，并驱动 SeedVR2 7B / 3B 模型分块调度。

## 安装方法 (Installation)

### 方式 1：进入 ComfyUI/custom_nodes 目录 Git Clone
`ash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/qiushengdesing/ComfyUI-ERP-SeamTools.git
pip install -r ComfyUI-ERP-SeamTools/requirements.txt
`

### 方式 2：使用 ComfyUI Manager
通过 Manager 的 Install via Git URL 输入此仓库地址即可一键安装。

## 节点列表 (Nodes Included)
- Shaniu6NativeSeamHeal (山牛6.0全景无缝接缝修复)
- ERPOutputResolutionv3 (全景输出分辨率规划器)
- ERPPrepare360v2 (ERP全景预处理)
- ERPReview360v2 (ERP全景质量审查与接缝校验)
- ERPSeamToneRepair (全景边缘色调谐波融合)
- ERPSeedVRPlanv3 (ERP + SeedVR2 专用显存与分块配置)
- ERPWrapPadv2 (ERP循环边缘填充)
- ERPWrapRestorev2 (ERP边缘裁切与无损还原)
- PanoramaFlux2DualViewModelPrototype (全景双视角协同)
- PanoramaFlux2WrapVAEDecodePrototype (全景循环 VAE 解码)
- PanoramaFlux2WrapVAEEncodePrototype (全景循环 VAE 编码)
- R2PStillUpscaleControls (单张图放大长短边控制枢纽)
