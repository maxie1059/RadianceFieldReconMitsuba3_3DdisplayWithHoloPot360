import os
import re
import gc
import drjit as dr
import mitsuba as mi
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------
class RadianceFieldPRB(mi.python.ad.integrators.common.RBIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.bbox     = mi.ScalarBoundingBox3f([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.use_relu = props.get('use_relu',True)
        self.grid_res = props.get('grid_init_res',16)
        self.sh_degree = props.get('sh_degree',2)
        res = self.grid_res
        self.sigmat    = mi.Texture3f(dr.full(mi.TensorXf, 0.01,
                                              shape=(self.grid_res, self.grid_res, self.grid_res, 1)))
        self.sh_coeffs = mi.Texture3f(dr.full(mi.TensorXf, 0.1,
                                              shape=(self.grid_res, self.grid_res, self.grid_res,
                                                     3 * (self.sh_degree + 1) ** 2)))

    def eval_emission(self, pos, direction):
        spec         = mi.Spectrum(0)
        sh_dir_coef  = dr.sh_eval(direction, self.sh_degree)
        sh_coeffs    = self.sh_coeffs.eval(pos)
        for i, sh in enumerate(sh_dir_coef):
            spec += sh * mi.Spectrum(sh_coeffs[3 * i:3 * (i + 1)])
        return dr.clip(spec, 0.0, 1.0)

    @dr.syntax
    def sample(self, mode, scene, sampler,
               ray, δL, state_in, active, **kwargs):
        primal = mode == dr.ADMode.Primal

        ray  = mi.Ray3f(ray)
        hit, mint, maxt = self.bbox.ray_intersect(ray)

        active  = mi.Bool(active)
        active &= hit
        if not primal:
            active &= dr.any(δL != 0)

        step_size = mi.Float(1.0 / self.grid_res)
        t  = mi.Float(mint) + sampler.next_1d(active) * step_size
        L  = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        β  = mi.Spectrum(1.0)

        while active:
            p = ray(t)
            with dr.resume_grad(when=not primal):
                sigmat = self.sigmat.eval(p)[0]
                if self.use_relu:
                    sigmat = dr.maximum(sigmat, 0.0)
                tr = dr.exp(-sigmat * step_size)
                Le = β * (1.0 - tr) * self.eval_emission(p, ray.d)

            β *= tr
            L  = L + Le if primal else L - Le

            with dr.resume_grad(when=not primal):
                if not primal:
                    dr.backward_from(δL * (L * tr / dr.detach(tr) + Le))

            t     += step_size
            active &= (t < maxt) & dr.any(β != 0.0)

        return L if primal else δL, mi.Bool(True), [], L

    def traverse(self, cb):
        cb.put("sigmat",    self.sigmat.tensor(),    mi.ParamFlags.Differentiable)
        cb.put('sh_coeffs', self.sh_coeffs.tensor(), mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        self.sigmat.update_inplace()
        self.sh_coeffs.update_inplace()
        self.grid_res = self.sigmat.shape[0]

mi.register_integrator("rf_prb", lambda props: RadianceFieldPRB(props))

class NeRF:
    def __init__(self, training_dir,
                 width=468, 
                 height=936, 
                 num_stages=4,
                 num_iterations_per_stage=15,
                 learning_rate=0.2,
                 grid_init_res=16,
                 sh_degree=2,
                 use_relu=True,
                 CHECKPOINT_EVERY=6,
                 sensor_width=17.3, #mm
                 focal_length=20.0): #mm
        
        # ---------------------------------------------------------------------------
        # Hyperparameters
        # ---------------------------------------------------------------------------
        self.width                    = width
        self.height                   = height
        self.num_stages               = num_stages
        self.num_iterations_per_stage = num_iterations_per_stage
        self.learning_rate            = learning_rate
        self.grid_init_res            = grid_init_res
        self.sh_degree                = sh_degree
        self.use_relu                 = use_relu

        # How often (in iterations) to write a checkpoint.  Set to 1 to save every
        # iteration; increase to reduce I/O overhead for fast GPUs.
        self.CHECKPOINT_EVERY = CHECKPOINT_EVERY

        self.sensor_width = sensor_width #MicroFourThirds standard, value in mm
        self.focal_length = focal_length

        # Preparing checkpoints directory
        self.CHECKPOINT_DIR = "checkpoints"
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

        # Set training dataset directory
        self.training_dir = training_dir

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------
    def plot_list(self,images, title=None):
        fig, axs = plt.subplots(1, len(images), figsize=(18, 3))
        
        if len(images) > 1:
            for i, img in enumerate(images):
                img_clipped = np.clip(img, 0.0, 1.0)          # clamp any values slightly outside [0,1]
                img_uint8   = (img_clipped * 255).astype(np.uint8)
                axs[i].imshow(mi.util.convert_to_bitmap(img_uint8))
                axs[i].axis('off')
        else:
            img_clipped = np.clip(images[0], 0.0, 1.0)          # clamp any values slightly outside [0,1]
            img_uint8   = (img_clipped * 255).astype(np.uint8)
            axs.imshow(mi.util.convert_to_bitmap(img_uint8))
            axs.axis('off')
        if title is not None:
            plt.suptitle(title)
        plt.show()

    # ---------------------------------------------------------------------------
    # Checkpoint helpers
    # ---------------------------------------------------------------------------

    def checkpoint_path(self,stage, iteration):
        """Return a combined string of checkpoint directory .npz for stage and iteration"""
        return os.path.join(self.CHECKPOINT_DIR, f"ckpt_s{stage:02d}_i{iteration:03d}.npz")

    def save_checkpoint(self,opt, stage, iteration, losses):
        """Save optimizer tensors + metadata to an .npz file."""
        path = self.checkpoint_path(stage, iteration)
        np.savez_compressed(
            path,
            sigmat=np.array(opt['sigmat']),
            sh_coeffs=np.array(opt['sh_coeffs']),
            stage=np.array(stage),
            iteration=np.array(iteration),
            losses=np.array(losses),
        )
        print(f"\n  [checkpoint saved] {path}")

    def find_latest_checkpoint(self,CHECKPOINT_DIR):
        """Return the path of the most recently written checkpoint, or None."""
        files = [
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith("ckpt_") and f.endswith(".npz")
        ]
        if not files:
            return None
        # Sort lexicographically — stage then iteration both zero-padded
        files.sort()
        return os.path.join(CHECKPOINT_DIR, files[-1])

    def load_checkpoint(self, path):
        """
        Restore integrator tensors directly at the saved resolution (no
        upsampling), then rebuild params/opt to match.
        Returns (stage, iteration, losses).
        """
        data      = np.load(path)
        saved_res = data['sigmat'].shape[0]

        self.integrator.sigmat    = mi.Texture3f(mi.TensorXf(data['sigmat']))
        self.integrator.sh_coeffs = mi.Texture3f(mi.TensorXf(data['sh_coeffs']))
        self.integrator.grid_res  = saved_res

        self.params = mi.traverse(self.integrator)
        self.opt    = mi.ad.Adam(lr=self.learning_rate,
                                 params={'sigmat':    self.params['sigmat'],
                                         'sh_coeffs': self.params['sh_coeffs']})
        self.params.update(self.opt)

        stage     = int(data['stage'])
        iteration = int(data['iteration'])
        losses    = list(data['losses'])
        print(f"[checkpoint loaded] {path}  (stage={stage}, iter={iteration}, grid={saved_res}^3)")
        return stage, iteration, losses

    def prep(self):
        """
        Prepare for training:
        1. Load reference images from directory
        2. Init Mitsuba3 sensors, scene
        """
        # ---------------------------------------------------------------------------
        # Load reference images  (kept as float32 CPU arrays)
        # ---------------------------------------------------------------------------

        root = self.training_dir
        directory = os.fsencode(root)

        files = sorted(
            os.listdir(directory),
            key=lambda f: int(re.search(r'\d+', os.fsdecode(f)).group())
        )

        # Store as CPU numpy arrays — only move to GPU one at a time during training.
        self.ref_images_np = []
        for file in files:
            img_name = os.fsdecode(file)
            fileDir  = os.path.join(root, img_name)
            img_bgr  = cv2.imread(fileDir, cv2.IMREAD_COLOR)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.ref_images_np.append(np.array(img_rgb, dtype=np.float32) / 255.0)

        self.sensor_count = len(self.ref_images_np)
        print(f"Loaded {self.sensor_count} reference images (CPU only)")

        # ---------------------------------------------------------------------------
        # Sensors
        # ---------------------------------------------------------------------------
        
        fov_rad = 2 * np.atan(self.sensor_width / (2 * self.focal_length))
        fov_deg = np.degrees(fov_rad)

        self.sensors = []
        for i in range(self.sensor_count):
            if self.sensor_count == 1:
                angle = 0.0
            else:
                start = -360.0/2
                step = 360.0 / (self.sensor_count-1)
                angle = start + i*step
            #angle = 360.0 / self.sensor_count * i
            self.sensors.append(mi.load_dict({
                'type': 'perspective',
                'fov': fov_deg,
                'to_world': mi.ScalarTransform4f() \
                            .translate([0.5, 0.5, 0.5]) \
                            .rotate([0, 1, 0], angle) \
                            .look_at(target=[0, 0, 0],
                                    origin=[0, 0, 1.3],
                                    up=[0, 1, 0]),
                
                'film': {
                    'type': 'hdrfilm',
                    'width': self.width,
                    'height': self.height,
                    'filter': {'type': 'box'},
                    'pixel_format': 'rgb',
                }
            }))

        self.scene      = mi.load_dict({'type': 'scene',
                                'integrator': {'type':          'rf_prb',
                                            'use_relu':      self.use_relu,
                                            'grid_init_res': self.grid_init_res,
                                            'sh_degree':     self.sh_degree},
                                'emitter':    {'type': 'constant'}})
        self.integrator = self.scene.integrator()
    
    def resume_last_ckpt(self):
        """
        Resume from last checkpoint.
        1. Init integrator parameters and optimizer
        2. If a checkpoint exists, restore tensors directly at their saved
           resolution (no upsampling needed — the .npz already contains the
           correct grid size for that stage).
        3. Advance resume_stage / resume_iter so train() knows where to continue.
        """
        self.resume_stage = 0
        self.resume_iter  = 0
        self.losses       = []

        latest_ckpt = self.find_latest_checkpoint(self.CHECKPOINT_DIR)

        if latest_ckpt:
            # --- Load raw numpy arrays from disk first (no GPU allocation yet) ---
            data      = np.load(latest_ckpt)
            saved_res = data['sigmat'].shape[0]   # actual resolution in the file

            # Build the integrator tensors at the saved resolution directly,
            # bypassing the default grid_init_res so we never allocate the
            # small grid and upsample — that intermediate step is what OOMs.
            self.integrator.sigmat = mi.Texture3f(mi.TensorXf(data['sigmat']))
            self.integrator.sh_coeffs = mi.Texture3f(mi.TensorXf(data['sh_coeffs']))
            self.integrator.grid_res = saved_res

            self.params = mi.traverse(self.integrator)
            self.opt    = mi.ad.Adam(lr=self.learning_rate,
                                     params={'sigmat':    self.params['sigmat'],
                                             'sh_coeffs': self.params['sh_coeffs']})
            self.params.update(self.opt)

            self.resume_stage = int(data['stage'])
            self.resume_iter  = int(data['iteration'])
            self.losses       = list(data['losses'])

            print(f"[checkpoint loaded] {latest_ckpt}  "
                  f"(stage={self.resume_stage}, iter={self.resume_iter}, "
                  f"grid={saved_res}³)")

            # Advance past the saved position
            if self.resume_iter >= self.num_iterations_per_stage - 1:
                self.resume_stage += 1
                self.resume_iter   = 0
            else:
                self.resume_iter  += 1

            print(f"Resuming from stage {self.resume_stage}, iteration {self.resume_iter}")

        else:
            # Fresh start — build params/opt from the default-initialised integrator
            self.params = mi.traverse(self.integrator)
            self.opt    = mi.ad.Adam(lr=self.learning_rate,
                                     params={'sigmat':    self.params['sigmat'],
                                             'sh_coeffs': self.params['sh_coeffs']})
            self.params.update(self.opt)
            print("No checkpoint found — starting from scratch")

    def train(self):
        """
        1. Resume from last checkpoint (if any)
        Train to get best optimization parameters
        """
        self.resume_last_ckpt()
        intermediate_images = []

        for stage in range(self.resume_stage, self.num_stages):
            print(f"\nStage {stage+1:02d}, grid resolution -> {self.opt['sigmat'].shape[0]}")

            start_iter = self.resume_iter if stage == self.resume_stage else 0

            for it in range(start_iter, self.num_iterations_per_stage):
                total_loss     = 0.0
                stage_end_imgs = []          # only populated on the final iteration

                for sensor_idx in range(self.sensor_count):
                    # --- Upload one ref image to GPU, use it, then release it -------
                    ref_gpu = mi.TensorXf(self.ref_images_np[sensor_idx])

                    img  = mi.render(self.scene, self.params,
                                    sensor=self.sensors[sensor_idx], spp=1, seed=it)
                    loss = dr.mean(dr.abs(img - ref_gpu), axis=None)
                    dr.backward(loss)

                    # Collect scalar loss before releasing the graph
                    dr.eval(loss)
                    total_loss += float(loss.array[0])

                    if it == self.num_iterations_per_stage - 1:
                        dr.eval(img)
                        # Keep as CPU numpy to free GPU memory immediately
                        stage_end_imgs.append(np.array(img))

                    # Release GPU tensors for this sensor immediately
                    del img, loss, ref_gpu
                    dr.flush_malloc_cache()

                self.losses.append(total_loss)
                self.opt.step()

                if not self.integrator.use_relu:
                    self.opt['sigmat'] = dr.maximum(self.opt['sigmat'], 0.0)

                self.params.update(self.opt)
                print(f"  --> iteration {it+1:02d}: error={total_loss:.6f}", end='\r')

                # --- Checkpoint ------------------------------------------------------
                if (it + 1) % self.CHECKPOINT_EVERY == 0 or it == self.num_iterations_per_stage - 1:
                    self.save_checkpoint(self.opt, stage, it, self.losses)

                # Flush Python GC + DrJit malloc cache every iteration
                gc.collect()
                dr.flush_malloc_cache()

            # if stage_end_imgs:
            #     intermediate_images.append(stage_end_imgs) # can be used to draw last iterations from each stage
            #     # Comment this out to plot those immediate_images
            #     for stage, inter in enumerate(intermediate_images):
            #         self.plot_list(inter, f'Stage {stage}')


            # Upsample for the next stage
            if stage < self.num_stages - 1:
                new_res   = 2 * self.opt['sigmat'].shape[0]
                new_shape = [new_res, new_res, new_res]
                self.opt['sigmat']    = dr.upsample(self.opt['sigmat'],    new_shape)
                self.opt['sh_coeffs'] = dr.upsample(self.opt['sh_coeffs'], new_shape)
                self.params.update(self.opt)
                dr.flush_malloc_cache()

        print('\nDone')
    
    def loss_plot(self):
        """Plotting the losses through all iterations"""
        title = "Training Loss " + str(self.sensor_count) + " views"

        plt.plot(self.losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def render_final_images(self):
        """
        Render and plot final images with the latest optimized parameters.
        MUST call .prep() before running this. Automatically loads the latest
        checkpoint at its saved resolution — no upsampling, no OOM.
        """
        self.resume_last_ckpt()
        final_images = [mi.render(self.scene, sensor=self.sensors[i], spp=128) for i in range(self.sensor_count)]
        self.plot_list(final_images, 'Final')

    def render_last_ckpt(self, CHECKPOINT_DIR=None):
        """
        Skip training entirely and just render from the latest checkpoint.
        Tensors are restored at their saved resolution — no upsampling needed.
        MUST call .prep() before running this.
        """
        ckpt_dir = CHECKPOINT_DIR or self.CHECKPOINT_DIR
        latest = self.find_latest_checkpoint(ckpt_dir)
        if latest is None:
            print('No checkpoint found.')
            return
        self.load_checkpoint(latest)
        final_images = [mi.render(self.scene, sensor=self.sensors[i], spp=128)
                      for i in range(self.sensor_count)]
        self.plot_list(final_images, 'Final (from checkpoint)')

