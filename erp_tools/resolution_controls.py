"""One selectable output size drives both SeedVR and ERP restoration.

No image tensors or model state are changed here. SeedVR's resolution is the
short side, so a final whole-sphere ERP of W x W/2 needs resolution=W/2.
The workflow must keep max_resolution=0 on SeedVR and connect BOTH outputs.
"""


WIDTH_OPTIONS = (
    "4500", "5000", "6000", "7000", "8000",
    "10000", "12000", "14000", "16000", "20000", "36000"
)


class ERPOutputResolutionv3:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "output_width": (list(WIDTH_OPTIONS), {
                "default": "4500",
                "tooltip": "最终全景宽度；高度自动取一半，并联动放大与整圈还原。",
            }),
        }}

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("final_width", "seedvr_short_side", "report")
    FUNCTION = "select"
    CATEGORY = "panorama/validated-sizing"
    DESCRIPTION = "选择最终全景尺寸，同步控制 SeedVR 短边与全景还原宽度。"

    def select(self, output_width="4500"):
        if isinstance(output_width, bool) or str(output_width) not in WIDTH_OPTIONS:
            raise ValueError(f"最终全景宽度请选择: {', '.join(WIDTH_OPTIONS)}。")
        width = int(output_width)
        height = width // 2
        report = (
            f"最终输出 {width}×{height}，完整 2:1 全景。"
            f"已同时设置放大短边 {height} 与整圈还原宽度 {width}。"
            "环绕补边比例由流程配置决定，中间图包含额外上下文；"
            "其具体尺寸以放大配置与还原报告为准，不能作为最终全景上传。"
            "放大节点 max_resolution 必须保持 0；更大尺寸需要更多内存和时间，"
            "增加输出像素不代表原本缺失的细节一定能恢复。"
        )
        return width, height, report


NODE_CLASS_MAPPINGS = {"ERPOutputResolutionv3": ERPOutputResolutionv3}
NODE_DISPLAY_NAME_MAPPINGS = {"ERPOutputResolutionv3": "全景输出分辨率 · 联动选择 v3"}
