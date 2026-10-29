"""Preserve source geometry at ERP poles instead of hallucinating stretched pixels."""
import math
import torch
import torch.nn.functional as F

class ERPPoleSourceGuardv2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'reference':('IMAGE',),'image':('IMAGE',),
                            'mode':(['保留原图极点结构','关闭'],)}}
    RETURN_TYPES=('IMAGE','STRING')
    RETURN_NAMES=('image','report')
    FUNCTION='protect'
    CATEGORY='panorama/validated-sizing'
    def protect(self,reference,image,mode='保留原图极点结构'):
        for name,a in (('原图',reference),('结果',image)):
            if a.ndim!=4 or a.shape[-1]!=3 or a.shape[2]!=2*a.shape[1] or not torch.isfinite(a).all().item():
                raise ValueError(name+'必须是有限RGB的精确2:1完整全景。')
        if reference.shape[0]!=image.shape[0]:raise ValueError('原图和结果数量不一致。')
        if mode=='关闭':return image,'极点保护已关闭。'
        if mode!='保留原图极点结构':raise ValueError('未知极点保护模式。')
        h,w=image.shape[1:3]
        ref=F.interpolate(reference.movedim(-1,1).float(),size=(h,w),mode='bicubic',align_corners=False,antialias=True).movedim(1,-1).clamp(0,1).to(image)
        latitude=(90-(torch.arange(h,device=image.device,dtype=torch.float32)+.5)*180/h).abs()
        # Match each pole's broad exposure/color, never its invented texture.
        # A constant RGB offset per pole preserves source line positions and
        # detail contrast; robust medians avoid isolated lamps/reflections.
        cap=max(1,round(h*25/180))
        adjusted=ref.clone()
        for begin,end in ((0,h//2),(h//2,h)):
            region=slice(0,cap) if begin==0 else slice(h-cap,h)
            source_small=F.interpolate(ref[:,region].movedim(-1,1).float(),size=(32,128),mode='area')
            target_small=F.interpolate(image[:,region].movedim(-1,1).float(),size=(32,128),mode='area')
            delta=(target_small.flatten(2).median(-1).values-source_small.flatten(2).median(-1).values).clamp(-.12,.12)
            adjusted[:,begin:end]=(ref[:,begin:end]+delta[:,None,None,:].to(ref)).clamp(0,1)
        # Keep the complete polar perspective center; most of the sphere
        # (abs(latitude)<45) remains exactly the generated image.
        t=((latitude-45)/20).clamp(0,1)
        mask=(t*t*(3-2*t))[None,:,None,None]
        result=torch.lerp(image,adjusted,mask.to(image)).clamp(0,1)
        return result,('极点结构保护：正上方和正下方65°以外采用原图结构与细节，并温和匹配本张输出的整体明暗色调，45°—65°平滑过渡；'
                       '其余区域保持洗图结果。此步骤不生成新物体、不改变原图极点几何，也不修复原图已有缺陷。'
                       '极点锐度与原始全景分辨率有关；原图水印或旧纹理也会保留。仍需检查过渡区域。')

NODE_CLASS_MAPPINGS={'ERPPoleSourceGuardv2':ERPPoleSourceGuardv2}
NODE_DISPLAY_NAME_MAPPINGS={'ERPPoleSourceGuardv2':'360全景 · 保留原图极点结构 v2'}
