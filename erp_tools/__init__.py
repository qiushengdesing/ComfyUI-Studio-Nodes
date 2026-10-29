"""Local panorama seam utilities, separate from existing workflows and weights."""
from pathlib import Path
import json, os, subprocess, sys, tempfile, time
import numpy as np
from PIL import Image
import torch
import folder_paths
import comfy.model_management
from .wrap_sampling import NODE_CLASS_MAPPINGS as WRAP_NODES
from .wrap_sampling import NODE_DISPLAY_NAME_MAPPINGS as WRAP_NAMES


class ERPSeamToneRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',),
            'mode': (['auto', 'force', 'bypass'],),
            'repair_band_percent': ('FLOAT', {'default': 9.0, 'min': 1.0, 'max': 20.0, 'step': 0.5}),
            'fine_line_cleanup': ('BOOLEAN', {'default': True}),
            'cleanup_width_at_4500px': ('INT', {'default': 16, 'min': 4, 'max': 32, 'step': 2}),
        }}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('repaired', 'join_at_center', 'report')
    FUNCTION = 'repair'
    CATEGORY = 'panorama/seam'

    def repair(self, image, mode='auto', repair_band_percent=9.0,
               fine_line_cleanup=True, cleanup_width_at_4500px=16):
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError('需要 RGB IMAGE 输入。')
        height, width = image.shape[1:3]
        if abs(width-2*height)>2 or width<256:
            raise ValueError('此节点用于 2:1 的完整360度全景，不能直接输入普通照片或全景查看器截图。')
        if mode == 'bypass':
            return image, torch.roll(image, width//2, 2), '已旁路，图像原样返回。'
        results=[];reports=[]
        model=Path(folder_paths.models_dir)/'LaMa/big-lama.pt'
        with tempfile.TemporaryDirectory(prefix='comfy_erp_seam_') as temp:
            temp=Path(temp)
            for index, frame in enumerate(image):
                comfy.model_management.throw_exception_if_processing_interrupted()
                source=temp/f'{index}_source.png';output=temp/f'{index}_result.png'
                summary=temp/f'{index}_report.json'
                Image.fromarray(np.rint(frame.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8)).save(source)
                command=[sys.executable,'-X','utf8','-s',str(Path(__file__).with_name('repair_worker.py')),
                    '--source',str(source),'--output',str(output),'--report',str(summary),
                    '--model',str(model),'--mode',mode,'--band',str(repair_band_percent/100),
                    '--cleanup',str(int(fine_line_cleanup)), '--width',str(cleanup_width_at_4500px)]
                environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=''
                with (temp/'worker.log').open('w',encoding='utf8') as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    try:
                        while process.poll() is None:
                            time.sleep(.25)
                            comfy.model_management.throw_exception_if_processing_interrupted()
                        if process.returncode:
                            log.flush()
                            raise RuntimeError((temp/'worker.log').read_text(encoding='utf8')[-5000:])
                    finally:
                        if process.poll() is None:
                            process.terminate();process.wait(timeout=10)
                info=json.loads(summary.read_text(encoding='utf8'));reports.append(info)
                if info['skipped']:
                    results.append(frame.clone());continue
                repaired=torch.from_numpy(np.array(Image.open(output).convert('RGB'))).to(frame).div(255)
                # Keep all original floating-point samples outside the repair band.
                margin=min(width//2,round(width*repair_band_percent/100))
                repaired[:,margin:-margin]=frame[:,margin:-margin]
                results.append(repaired)
        result=torch.stack(results)
        return result, torch.roll(result,width//2,2), json.dumps(reports,ensure_ascii=False,indent=2)


NODE_CLASS_MAPPINGS = {**WRAP_NODES, 'ERPSeamToneRepair': ERPSeamToneRepair}
NODE_DISPLAY_NAME_MAPPINGS = {**WRAP_NAMES, 'ERPSeamToneRepair': '360全景 · 色差与细线修缝（结构需检查）'}

from .pipeline_v2 import NODE_CLASS_MAPPINGS as PIPELINE_V2_NODES, NODE_DISPLAY_NAME_MAPPINGS as PIPELINE_V2_NAMES
NODE_CLASS_MAPPINGS.update(PIPELINE_V2_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(PIPELINE_V2_NAMES)

from .quality_preview import NODE_CLASS_MAPPINGS as REVIEW_V2_NODES, NODE_DISPLAY_NAME_MAPPINGS as REVIEW_V2_NAMES
NODE_CLASS_MAPPINGS.update(REVIEW_V2_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(REVIEW_V2_NAMES)

from .pole_guard import NODE_CLASS_MAPPINGS as POLE_V2_NODES, NODE_DISPLAY_NAME_MAPPINGS as POLE_V2_NAMES
NODE_CLASS_MAPPINGS.update(POLE_V2_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(POLE_V2_NAMES)

from .resolution_controls import NODE_CLASS_MAPPINGS as RESOLUTION_V3_NODES, NODE_DISPLAY_NAME_MAPPINGS as RESOLUTION_V3_NAMES
NODE_CLASS_MAPPINGS.update(RESOLUTION_V3_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(RESOLUTION_V3_NAMES)

from .pole_guard_v3 import NODE_CLASS_MAPPINGS as POLE_V3_NODES, NODE_DISPLAY_NAME_MAPPINGS as POLE_V3_NAMES
NODE_CLASS_MAPPINGS.update(POLE_V3_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(POLE_V3_NAMES)

from .seedvr_plan_v3 import NODE_CLASS_MAPPINGS as SEEDVR_PLAN_V3_NODES, NODE_DISPLAY_NAME_MAPPINGS as SEEDVR_PLAN_V3_NAMES
NODE_CLASS_MAPPINGS.update(SEEDVR_PLAN_V3_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SEEDVR_PLAN_V3_NAMES)

from .fidelity_nodes import NODE_CLASS_MAPPINGS as FIDELITY_V4_NODES, NODE_DISPLAY_NAME_MAPPINGS as FIDELITY_V4_NAMES
NODE_CLASS_MAPPINGS.update(FIDELITY_V4_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(FIDELITY_V4_NAMES)

from .single_ceiling_guard import NODE_CLASS_MAPPINGS as SINGLE_CEILING_NODES, NODE_DISPLAY_NAME_MAPPINGS as SINGLE_CEILING_NAMES
NODE_CLASS_MAPPINGS.update(SINGLE_CEILING_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SINGLE_CEILING_NAMES)

from .erp_full_ceiling_guard import NODE_CLASS_MAPPINGS as FULL_CEILING_NODES, NODE_DISPLAY_NAME_MAPPINGS as FULL_CEILING_NAMES
NODE_CLASS_MAPPINGS.update(FULL_CEILING_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(FULL_CEILING_NAMES)

from .smart_graffiti_guard import NODE_CLASS_MAPPINGS as SMART_GRAFFITI_NODES, NODE_DISPLAY_NAME_MAPPINGS as SMART_GRAFFITI_NAMES
NODE_CLASS_MAPPINGS.update(SMART_GRAFFITI_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SMART_GRAFFITI_NAMES)

from .smart_seed_guard import NODE_CLASS_MAPPINGS as SMART_SEED_NODES, NODE_DISPLAY_NAME_MAPPINGS as SMART_SEED_NAMES
NODE_CLASS_MAPPINGS.update(SMART_SEED_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SMART_SEED_NAMES)

from .shaniu_v6_native import NODE_CLASS_MAPPINGS as SHANIU_V6_NODES, NODE_DISPLAY_NAME_MAPPINGS as SHANIU_V6_NAMES
NODE_CLASS_MAPPINGS.update(SHANIU_V6_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SHANIU_V6_NAMES)




try:
    from .r2p_single_image_controls import NODE_CLASS_MAPPINGS as R2P_CONTROLS_NODES, NODE_DISPLAY_NAME_MAPPINGS as R2P_CONTROLS_NAMES
    NODE_CLASS_MAPPINGS.update(R2P_CONTROLS_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(R2P_CONTROLS_NAMES)
except Exception as e:
    print(f'[ComfyUI-ERP-SeamTools] Note: r2p_single_image_controls skipped: {e}')
