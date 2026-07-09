# Register a custom integrator that only keeps radiance from rays that hit the slab
import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

class SlabMaskedIntegrator(mi.SamplingIntegrator):
    """
    Path integrator that only returns radiance for rays that pass through
    the bounding box of the slab shape. All other rays return black.
    """
    def __init__(self, props):
        super().__init__(props)
        # AABB of the slab in world space — set from glass_size
        # slab is a cube scaled by glass_size = [4, 6, 0.1], rotated 90° around Y
        # After rotation: X extent = glass_size[1], Y extent = glass_size[2], Z extent = glass_size[0]
        hw = props.get('half_width',  6.0)   # glass_size[1]
        hh = props.get('half_height', 4.0)   # glass_size[0]
        ht = props.get('half_thick',  0.1)   # glass_size[2]
        self.bbox_min = mi.ScalarPoint3f(-ht, -hw, -hh)
        self.bbox_max = mi.ScalarPoint3f( ht,  hw,  hh)
        self.path = mi.load_dict({'type': 'path', 'max_depth': -1, 'hide_emitters': False})

    def sample(self, scene, sampler, ray, medium=None, active=True):
        # --- 1. Ray-AABB intersection test ---
        inv_d = dr.rcp(ray.d)
        t_min_vec = (self.bbox_min - ray.o) * inv_d
        t_max_vec = (self.bbox_max - ray.o) * inv_d
        t_enter = dr.max(dr.minimum(t_min_vec, t_max_vec))  # component-wise max of mins
        t_exit  = dr.min(dr.maximum(t_min_vec, t_max_vec))  # component-wise min of maxes
        hits_slab = (t_exit >= t_enter) & (t_exit >= 0.0)

        # --- 2. Only run full path tracing where ray hits slab ---
        active_masked = mi.Bool(active) & hits_slab
        Li, valid, aovs = self.path.sample(scene, sampler, ray, medium, active_masked)

        # --- 3. Zero out rays that missed the slab ---
        Li[~hits_slab] = mi.Color3f(0.0)

        return Li, valid, aovs

mi.register_integrator("slab_masked_path", lambda props: SlabMaskedIntegrator(props))