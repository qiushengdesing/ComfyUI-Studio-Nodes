"""Pure NumPy conservative seam gate, independent of Comfy, torch and CUDA.
This is an anomaly detector, not proof that image content needs reconstruction.
A False result means return the input unchanged. Geometry should default OFF.
"""
import numpy as np

def should_repair_seam(image, absolute_threshold=3.0, ratio_threshold=2.5, coherent_bias_threshold=4.0, allow_gradient_only=False):
    original=np.asarray(image)
    if original.ndim!=3 or original.shape[2]<3:raise ValueError('Expected HxWxRGB array')
    f=original[...,:3].astype(np.float32)
    if not np.isfinite(f).all():raise ValueError('Image contains NaN or infinity')
    if np.issubdtype(original.dtype,np.floating) and float(f.max(initial=0))<=1.001:f=f*255.0
    h,w=f.shape[:2]
    if h<8 or w<16:raise ValueError('Image is too small for seam detection')
    n=min(32,max(8,w//64))
    jump=f[:,0]-f[:,-1]
    sides=np.concatenate([f[:,1:n]-f[:,:n-1],f[:,-n+1:]-f[:,-n:-1]],axis=1)
    jump_mae=float(np.abs(jump).mean())
    baseline=float(np.median(np.abs(sides).mean(axis=(0,2))))
    ratio=jump_mae/max(.25,baseline)
    coherent=np.median(jump,axis=0)-np.median(sides,axis=(0,1))
    coherent_magnitude=float(np.abs(coherent).mean())
    enabled=jump_mae>absolute_threshold and (coherent_magnitude>coherent_bias_threshold or (allow_gradient_only and ratio>ratio_threshold))
    details={'repair_recommended':bool(enabled),'gradient_only_suspected':bool(jump_mae>absolute_threshold and ratio>ratio_threshold and coherent_magnitude<=coherent_bias_threshold),'jump_rgb_mae_255':jump_mae,'neighbor_median_rgb_mae_255':baseline,'jump_ratio':ratio,'coherent_bias_rgb_255':coherent.tolist(),'coherent_bias_mean_abs_255':coherent_magnitude,'thresholds':{'minimum_jump':absolute_threshold,'minimum_ratio':ratio_threshold,'minimum_coherent_bias':coherent_bias_threshold},'reason':'Strong boundary anomaly' if enabled else 'Conservative skip: boundary is comparable to nearby image gradients','limitations':'Can miss weak/localized seams. A real sharp edge exactly at the wrap boundary can resemble an artifact. This does not certify geometric alignment.'}
    return bool(enabled),details

def allow_geometric_alignment(anchor_info,height):
    """Optional separate gate. Keep geometry disabled by default in the UI.
    Requires multiple high-score correspondences spread across the image height.
    """
    anchors=anchor_info.get('anchors',[])
    if not anchors:return False,{'reason':'No accepted anchors'}
    y=np.array([v['y'] for v in anchors]);d=np.array([v['seam_intercept'] for v in anchors])
    accepted=[r for r in anchor_info.get('records',[]) if r.get('accepted')]
    score=float(np.median([r['ncc'] for r in accepted])) if accepted else 0.0
    span=float((y.max()-y.min())/height);bins=len(set(np.minimum(7,(8*y/height).astype(int)).tolist()));shift=float(np.median(np.abs(d)))
    enabled=len(anchors)>=6 and span>=.30 and bins>=3 and score>=.80 and shift>=max(.75,height*.75/2250)
    return bool(enabled),{'anchor_count':len(anchors),'span_fraction':span,'occupied_bins_of_8':bins,'median_ncc':score,'median_abs_shift':shift,'reason':'Support threshold met; geometry remains experimental' if enabled else 'Insufficient reliable geometric support'}

