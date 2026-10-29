"""CPU process isolated from ComfyUI's CUDA and threading state."""
import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import torch
from tone_core import dual_harmonic
from lama_core import finish_panorama_lama_cpu

def main():
    parser=argparse.ArgumentParser()
    for name in ('source','output','report','model','mode'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--band',type=float,required=True)
    parser.add_argument('--cleanup',type=int,required=True)
    parser.add_argument('--width',type=int,required=True)
    args=parser.parse_args();started=time.perf_counter()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    image=np.array(Image.open(args.source).convert('RGB'))
    from seam_gate import should_repair_seam
    enabled,decision=should_repair_seam(image)
    decision['repair']=enabled
    report={'source_size':[image.shape[1],image.shape[0]],'gate':decision,
            'skipped':args.mode=='auto' and not decision['repair'],
            'scope':'色差和细竖线；不保证修复物体结构断裂。','geometry_warp':False}
    if report['skipped']:
        result=image
    else:
        result,tone=dual_harmonic(image,wide_fraction=args.band,narrow_width=max(2,round(image.shape[1]*6/4500)))
        report['tone']=tone
        if args.cleanup:
            model=Path(args.model)
            if not model.is_file():
                raise FileNotFoundError('缺少本地 LaMa 模型: '+str(model))
            size=image.shape[1]
            width=max(4,2*round(size*args.width/4500/2))
            context=min(768,size-size%8)
            result,cleanup=finish_panorama_lama_cpu(result,model_path=model,mask_width=width,
                context=context,overlap=min(128,context//4))
            report['cleanup']=cleanup
    report['runtime_seconds']=time.perf_counter()-started
    Image.fromarray(result).save(args.output)
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
