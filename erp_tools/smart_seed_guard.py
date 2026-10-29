"""Smart Auto Seed Controller for Two-Stage Workflows.
Automatically switches behavior based on whether upscale is active.
"""
import random
import time

class SmartWorkflowSeedGuard:
    last_seed = 4242
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mode': (['智能自动 (放大关=洗新图 / 放大开=锁定放大)', '保持固定 (fixed)', '每次随机 (randomize)'],),
            },
            'hidden': {
                'prompt': 'PROMPT',
                'unique_id': 'UNIQUE_ID'
            }
        }
    
    RETURN_TYPES = ('INT',)
    RETURN_NAMES = ('seed',)
    FUNCTION = 'get_seed'
    CATEGORY = 'sampling'

    @classmethod
    def IS_CHANGED(cls, mode, prompt=None, unique_id=None):
        if '保持固定' in mode:
            return cls.last_seed
        if '每次随机' in mode:
            return time.time()
        # Smart mode: if prompt contains upscale nodes, return cached seed!
        if prompt:
            upscale_nodes = {'506', '2010', '2015', '1010', '2101', '510'}
            is_upscaling = any(nid in prompt for nid in upscale_nodes)
            if is_upscaling:
                return cls.last_seed
        return time.time()

    def get_seed(self, mode, prompt=None, unique_id=None):
        if '保持固定' in mode:
            return (cls.last_seed,)
        
        is_upscaling = False
        if prompt:
            upscale_nodes = {'506', '2010', '2015', '1010', '2101', '510'}
            is_upscaling = any(nid in prompt for nid in upscale_nodes)
            
        if is_upscaling and '每次随机' not in mode:
            return (cls.last_seed,)
        else:
            cls.last_seed = random.randint(1, 1125899906842624)
            return (cls.last_seed,)

NODE_CLASS_MAPPINGS = {
    'SmartWorkflowSeedGuard': SmartWorkflowSeedGuard
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'SmartWorkflowSeedGuard': '🎲 智能洗图种子控制器 (关=洗新图 / 开=锁定放大)'
}
