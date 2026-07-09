"""
myNeRF_metrics.py
=================
Metrics, plots, and experiment runners.
Written to match the actual NeRF class in myNeRF_helper.py:
  - NeRF(training_dir, width, height, num_stages, num_iterations_per_stage, learning_rate, ...)
  - nerf.prep()   <- no arguments
  - nerf.train()  <- no arguments
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity  as _ssim
import drjit as dr
import mitsuba as mi


# ============================================================
# 1.  IMAGE QUALITY METRICS  -  PSNR and SSIM
# ============================================================

def compute_psnr(rendered: np.ndarray, reference: np.ndarray) -> float:
    return _psnr(reference, rendered, data_range=1.0)

def compute_ssim(rendered: np.ndarray, reference: np.ndarray) -> float:
    return _ssim(reference, rendered, data_range=1.0, channel_axis=2)

def evaluate_all_views(rendered_list, reference_list) -> dict:
    psnrs, ssims = [], []
    for r, ref in zip(rendered_list, reference_list):
        r   = np.clip(np.array(r,   dtype=np.float32), 0.0, 1.0)
        ref = np.clip(np.array(ref, dtype=np.float32), 0.0, 1.0)
        psnrs.append(compute_psnr(r, ref))
        ssims.append(compute_ssim(r, ref))
    return {
        'psnr_per_view': psnrs, 'ssim_per_view': ssims,
        'mean_psnr': float(np.mean(psnrs)), 'std_psnr': float(np.std(psnrs)),
        'mean_ssim': float(np.mean(ssims)), 'std_ssim': float(np.std(ssims)),
    }


# ============================================================
# 2.  PLOTTING
# ============================================================

def plot_loss_curves(loss_dict: dict, log_scale=True,
                     title='Training Loss', save_path=None):
    """loss_dict = {'label': [list of loss values], ...}"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for label, losses in loss_dict.items():
        x = range(1, len(losses) + 1)
        axes[0].plot(x, losses, label=label)
        axes[1].plot(x, losses, label=label)
    for ax, scale in zip(axes, ['Linear', 'Log']):
        ax.set_title(f'{title} - {scale} scale')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss' if scale == 'Linear' else 'Loss (log)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    axes[1].yaxis.set_minor_locator(ticker.LogLocator(subs='all'))
    axes[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_metrics_vs_views(view_counts, psnr_values, ssim_values, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(view_counts, psnr_values, 'o-', color='steelblue', linewidth=2)
    ax1.set_title('PSNR vs Number of Views')
    ax1.set_xlabel('Views')
    ax1.set_ylabel('Mean PSNR (dB)')
    ax1.grid(True, alpha=0.3)
    ax2.plot(view_counts, ssim_values, 'o-', color='darkorange', linewidth=2)
    ax2.set_title('SSIM vs Number of Views')
    ax2.set_xlabel('Views')
    ax2.set_ylabel('Mean SSIM')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_psnr_ssim_per_view(metrics: dict, label='', save_path=None):
    n = len(metrics['psnr_per_view'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.bar(range(n), metrics['psnr_per_view'], color='steelblue', alpha=0.8)
    ax1.axhline(metrics['mean_psnr'], color='red', linestyle='--',
                label=f"Mean {metrics['mean_psnr']:.2f} dB")
    ax1.set_title(f'Per-view PSNR {label}')
    ax1.set_xlabel('View index')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    ax2.bar(range(n), metrics['ssim_per_view'], color='darkorange', alpha=0.8)
    ax2.axhline(metrics['mean_ssim'], color='red', linestyle='--',
                label=f"Mean {metrics['mean_ssim']:.3f}")
    ax2.set_title(f'Per-view SSIM {label}')
    ax2.set_xlabel('View index')
    ax2.set_ylabel('SSIM')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_3d_surface(grid_x, grid_y, grid_z,
                    xlabel='Learning Rate', ylabel='Momentum (b1)',
                    zlabel='Mean PSNR (dB)', title='Hyperparameter Surface',
                    save_path=None):
    X, Y = np.meshgrid(grid_x, grid_y)
    fig  = plt.figure(figsize=(10, 6))
    ax   = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, grid_z, cmap='viridis', edgecolor='none', alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3.  SHARED INTERNAL HELPERS
# ============================================================

def _render_views(nerf) -> list:
    """Render all training sensors at spp=128 and return as numpy list."""
    return [np.array(mi.render(nerf.scene, sensor=nerf.sensors[i], spp=128))
            for i in range(nerf.sensor_count)]


# ============================================================
# 4.  EXPERIMENT 1 - LOSS FUNCTION COMPARISON  (L1 / L2 / Huber)
# ============================================================

def _build_train_loop(self_inner, loss_name):
    """Runs a full training loop on nerf using the named loss function.
    Called instead of nerf.train() so we can swap the loss without
    touching myNeRF_helper.py."""
    self_inner.resume_last_ckpt()
    for stage in range(self_inner.resume_stage, self_inner.num_stages):
        print(f"\nStage {stage+1:02d}, grid -> {self_inner.opt['sigmat'].shape[0]}")
        start_iter = self_inner.resume_iter if stage == self_inner.resume_stage else 0
        for it in range(start_iter, self_inner.num_iterations_per_stage):
            total_loss = 0.0
            for idx in range(self_inner.sensor_count):
                ref_gpu = mi.TensorXf(self_inner.ref_images_np[idx])
                img     = mi.render(self_inner.scene, self_inner.params,
                                    sensor=self_inner.sensors[idx], spp=1, seed=it)
                if loss_name == 'L1':
                    loss = dr.mean(dr.abs(img - ref_gpu), axis=None)
                elif loss_name == 'L2':
                    diff = img - ref_gpu
                    loss = dr.mean(diff * diff, axis=None)
                elif loss_name == 'Huber':
                    delta = 0.1
                    diff  = dr.abs(img - ref_gpu)
                    quad  = 0.5 * diff * diff
                    lin   = delta * (diff - 0.5 * delta)
                    loss  = dr.mean(dr.select(diff < delta, quad, lin), axis=None)
                dr.backward(loss); dr.eval(loss)
                total_loss += float(loss.array[0])
                del img, loss, ref_gpu; dr.flush_malloc_cache()

            self_inner.losses.append(total_loss)
            self_inner.opt.step()
            if not self_inner.integrator.use_relu:
                self_inner.opt['sigmat'] = dr.maximum(self_inner.opt['sigmat'], 0.0)
            self_inner.params.update(self_inner.opt)
            print(f"  --> it {it+1:02d}: loss={total_loss:.6f}", end='\r')
            if (it+1) % self_inner.CHECKPOINT_EVERY == 0 \
                    or it == self_inner.num_iterations_per_stage - 1:
                self_inner.save_checkpoint(self_inner.opt, stage, it, self_inner.losses)
            gc.collect(); dr.flush_malloc_cache()

        if stage < self_inner.num_stages - 1:
            new_res   = 2 * self_inner.opt['sigmat'].shape[0]
            new_shape = [new_res, new_res, new_res]
            self_inner.opt['sigmat']    = dr.upsample(self_inner.opt['sigmat'],    new_shape)
            self_inner.opt['sh_coeffs'] = dr.upsample(self_inner.opt['sh_coeffs'], new_shape)
            self_inner.params.update(self_inner.opt)
            dr.flush_malloc_cache()
    print('\nDone')


def run_loss_comparison(NeRFClass, training_dir: str,
                        width=500, height=500,
                        num_stages=4, num_iterations_per_stage=50,
                        learning_rate=0.2,
                        view_counts=36,
                        loss_names=None,
                        results_dir='results/loss_comparison'):
    """
    Train with L1, L2, Huber and compare loss curves + PSNR/SSIM.
    Each loss gets its own checkpoint folder inside results_dir so they
    never overwrite each other and you can resume any one individually.

    After it finishes, call plot_loss_comparison_from_checkpoints() to
    re-plot without re-training.
    """
    os.makedirs(results_dir, exist_ok=True)
    if loss_names is None:
        loss_names = ['L1', 'L2', 'Huber']

    all_losses = {}
    for loss_name in loss_names:
        print(f'\n{"="*50}\n  {loss_name} loss\n{"="*50}')
        nerf = NeRFClass(training_dir, width=width, height=height,
                         num_stages=num_stages,
                         num_iterations_per_stage=num_iterations_per_stage,
                         learning_rate=learning_rate)
        nerf.CHECKPOINT_DIR = os.path.join(results_dir, f'checkpoints_{loss_name}')
        os.makedirs(nerf.CHECKPOINT_DIR, exist_ok=True)
        nerf.prep()

        # Evenly spaced indices across the full 72 — preserves angular coverage
        indices = np.round(np.linspace(0, len(nerf.ref_images_np) - 1, view_counts)).astype(int)
        nerf.ref_images_np = [nerf.ref_images_np[i] for i in indices]
        nerf.sensors       = [nerf.sensors[i]       for i in indices]
        nerf.sensor_count  = view_counts

        _build_train_loop(nerf, loss_name)

        rendered = _render_views(nerf)
        metrics  = evaluate_all_views(rendered, nerf.ref_images_np)
        all_losses[loss_name] = nerf.losses

        np.savez(os.path.join(results_dir, f'summary_{loss_name}.npz'),
                 losses=nerf.losses,
                 mean_psnr=metrics['mean_psnr'],
                 mean_ssim=metrics['mean_ssim'])
        print(f'\n  {loss_name}  PSNR={metrics["mean_psnr"]:.2f} dB'
              f'  SSIM={metrics["mean_ssim"]:.4f}')
        del nerf, rendered; gc.collect()

    plot_loss_curves(all_losses, log_scale=True, title='Loss Comparison',
                     save_path=os.path.join(results_dir, 'loss_curves.png'))


def plot_loss_comparison_from_checkpoints(results_dir='results/loss_comparison',
                                          loss_names=None):
    """
    Re-plot the loss comparison WITHOUT re-training.
    Reads the summary_*.npz files written by run_loss_comparison().

    Usage (in myNeRF.py or a notebook):
        from myNeRF_metrics import plot_loss_comparison_from_checkpoints
        plot_loss_comparison_from_checkpoints('results/loss_comparison')
    """
    if loss_names is None:
        loss_names = ['L1', 'L2', 'Huber']
    all_losses = {}
    for name in loss_names:
        path = os.path.join(results_dir, f'summary_{name}.npz')
        if not os.path.exists(path):
            print(f'  [skip] {path} not found')
            continue
        data = np.load(path)
        all_losses[name] = list(data['losses'])
        print(f'  {name}  PSNR={float(data["mean_psnr"]):.2f} dB'
              f'  SSIM={float(data["mean_ssim"]):.4f}')
    plot_loss_curves(all_losses, log_scale=True, title='Loss Comparison')


# ============================================================
# 5.  EXPERIMENT 2 - VIEW COUNT EVOLUTION
# ============================================================

def run_view_count_experiment(NeRFClass, training_dir: str,
                              width=500, height=500,
                              num_stages=4, num_iterations_per_stage=50,
                              learning_rate=0.2,
                              view_counts=None,
                              results_dir='results/view_counts'):
    """
    Train with increasing numbers of views and record PSNR / SSIM.
    Your dataset must have at least max(view_counts) images.

    After it finishes, call plot_view_count_from_checkpoints() to re-plot.
    """
    os.makedirs(results_dir, exist_ok=True)
    if view_counts is None:
        view_counts = [9, 18, 36, 72]

    psnr_vals, ssim_vals = [], []
    for n_views in view_counts:
        print(f'\n{"="*50}\n  {n_views} views\n{"="*50}')
        nerf = NeRFClass(training_dir, width=width, height=height,
                         num_stages=num_stages,
                         num_iterations_per_stage=num_iterations_per_stage,
                         learning_rate=learning_rate)
        nerf.CHECKPOINT_DIR = os.path.join(results_dir, f'checkpoints_{n_views}views')
        os.makedirs(nerf.CHECKPOINT_DIR, exist_ok=True)
        nerf.prep()

        # Evenly spaced indices across the full 72 — preserves angular coverage
        indices = np.round(np.linspace(0, len(nerf.ref_images_np) - 1, n_views)).astype(int)
        nerf.ref_images_np = [nerf.ref_images_np[i] for i in indices]
        nerf.sensors       = [nerf.sensors[i]       for i in indices]
        nerf.sensor_count  = n_views

        nerf.train()

        rendered = _render_views(nerf)
        metrics  = evaluate_all_views(rendered, nerf.ref_images_np)
        psnr_vals.append(metrics['mean_psnr'])
        ssim_vals.append(metrics['mean_ssim'])

        np.savez(os.path.join(results_dir, f'summary_{n_views}views.npz'),
                 n_views=n_views, losses=nerf.losses,
                 mean_psnr=metrics['mean_psnr'],
                 mean_ssim=metrics['mean_ssim'])
        print(f'\n  {n_views} views  PSNR={metrics["mean_psnr"]:.2f} dB'
              f'  SSIM={metrics["mean_ssim"]:.4f}')
        del nerf, rendered; gc.collect()

    plot_metrics_vs_views(view_counts, psnr_vals, ssim_vals,
                          save_path=os.path.join(results_dir, 'metrics_vs_views.png'))


def plot_view_count_from_checkpoints(results_dir='results/view_counts',
                                     view_counts=None):
    """
    Re-plot PSNR/SSIM vs views WITHOUT re-training.

    Usage:
        from myNeRF_metrics import plot_view_count_from_checkpoints
        plot_view_count_from_checkpoints('results/view_counts', [7, 16, 32])
    """
    if view_counts is None:
        view_counts = [9, 18, 36, 72]
    psnr_vals, ssim_vals, found = [], [], []
    for n in view_counts:
        path = os.path.join(results_dir, f'summary_{n}views.npz')
        if not os.path.exists(path):
            print(f'  [skip] {path} not found')
            continue
        data = np.load(path)
        psnr_vals.append(float(data['mean_psnr']))
        ssim_vals.append(float(data['mean_ssim']))
        found.append(n)
        print(f'  {n} views  PSNR={float(data["mean_psnr"]):.2f} dB'
              f'  SSIM={float(data["mean_ssim"]):.4f}')
    plot_metrics_vs_views(found, psnr_vals, ssim_vals)


# ============================================================
# 6.  EXPERIMENT 3 - LEARNING RATE x MOMENTUM GRID SEARCH
# ============================================================

def run_lr_momentum_grid(NeRFClass, training_dir: str,
                         width=500, height=500,
                         num_stages=3, num_iterations_per_stage=30,
                         lr_values=None, momentum_values=None,
                         view_counts=36,
                         results_dir='results/hyperparam_grid'):
    """
    Grid search over learning_rate x Adam beta1 (momentum).
    num_stages=3 and num_iterations=30 kept low to limit runtime.
    After it finishes, call plot_lr_momentum_from_checkpoints() to re-plot.
    """
    os.makedirs(results_dir, exist_ok=True)
    if lr_values       is None: lr_values       = [0.05, 0.1, 0.2, 0.4]
    if momentum_values is None: momentum_values = [0.80, 0.85, 0.90, 0.95]

    psnr_grid = np.zeros((len(momentum_values), len(lr_values)))

    for j, lr in enumerate(lr_values):
        for i, mom in enumerate(momentum_values):
            print(f'\n{"="*50}\n  lr={lr}  beta1={mom}\n{"="*50}')
            nerf = NeRFClass(training_dir, width=width, height=height,
                             num_stages=num_stages,
                             num_iterations_per_stage=num_iterations_per_stage,
                             learning_rate=lr)
            nerf.CHECKPOINT_DIR = os.path.join(results_dir, f'ckpt_lr{lr}_mom{mom}')
            os.makedirs(nerf.CHECKPOINT_DIR, exist_ok=True)
            nerf.prep()

            # Evenly spaced indices across the full 72 — preserves angular coverage
            indices = np.round(np.linspace(0, len(nerf.ref_images_np) - 1, view_counts)).astype(int)
            nerf.ref_images_np = [nerf.ref_images_np[i] for i in indices]
            nerf.sensors       = [nerf.sensors[i]       for i in indices]
            nerf.sensor_count  = view_counts

            # resume_last_ckpt() builds self.opt with default beta1=0.9.
            # We call it manually here then patch beta1 before the loop runs.
            nerf.resume_last_ckpt()
            nerf.opt.beta_1 = mom   # patch Adam momentum in-place

            # Run the training loop directly (resume already done above)
            for stage in range(nerf.resume_stage, nerf.num_stages):
                print(f"\nStage {stage+1:02d}, grid -> {nerf.opt['sigmat'].shape[0]}")
                start_iter = nerf.resume_iter if stage == nerf.resume_stage else 0
                for it in range(start_iter, nerf.num_iterations_per_stage):
                    total_loss = 0.0
                    for idx in range(nerf.sensor_count):
                        ref_gpu = mi.TensorXf(nerf.ref_images_np[idx])
                        img     = mi.render(nerf.scene, nerf.params,
                                            sensor=nerf.sensors[idx], spp=1, seed=it)
                        loss = dr.mean(dr.abs(img - ref_gpu), axis=None)
                        dr.backward(loss); dr.eval(loss)
                        total_loss += float(loss.array[0])
                        del img, loss, ref_gpu; dr.flush_malloc_cache()
                    nerf.losses.append(total_loss)
                    nerf.opt.step()
                    if not nerf.integrator.use_relu:
                        nerf.opt['sigmat'] = dr.maximum(nerf.opt['sigmat'], 0.0)
                    nerf.params.update(nerf.opt)
                    print(f"  --> it {it+1:02d}: loss={total_loss:.6f}", end='\r')
                    if (it+1) % nerf.CHECKPOINT_EVERY == 0 \
                            or it == nerf.num_iterations_per_stage - 1:
                        nerf.save_checkpoint(nerf.opt, stage, it, nerf.losses)
                    gc.collect(); dr.flush_malloc_cache()

                if stage < nerf.num_stages - 1:
                    new_res   = 2 * nerf.opt['sigmat'].shape[0]
                    new_shape = [new_res, new_res, new_res]
                    nerf.opt['sigmat']    = dr.upsample(nerf.opt['sigmat'],    new_shape)
                    nerf.opt['sh_coeffs'] = dr.upsample(nerf.opt['sh_coeffs'], new_shape)
                    nerf.params.update(nerf.opt)
                    dr.flush_malloc_cache()
            print('\nDone')

            rendered = _render_views(nerf)
            metrics  = evaluate_all_views(rendered, nerf.ref_images_np)
            psnr_grid[i, j] = metrics['mean_psnr']

            np.savez(os.path.join(results_dir, f'summary_lr{lr}_mom{mom}.npz'),
                     lr=lr, mom=mom, losses=nerf.losses,
                     mean_psnr=metrics['mean_psnr'],
                     mean_ssim=metrics['mean_ssim'])
            print(f'\n  lr={lr} beta1={mom}  PSNR={metrics["mean_psnr"]:.2f} dB')
            del nerf, rendered; gc.collect()

    np.savez(os.path.join(results_dir, 'psnr_grid.npz'),
             lr_values=lr_values, momentum_values=momentum_values,
             psnr_grid=psnr_grid)
    plot_3d_surface(lr_values, momentum_values, psnr_grid,
                    save_path=os.path.join(results_dir, 'psnr_surface.png'))
    return psnr_grid


def plot_lr_momentum_from_checkpoints(results_dir='results/hyperparam_grid'):
    """
    Re-plot the 3-D surface WITHOUT re-training.
    Reads psnr_grid.npz written by run_lr_momentum_grid().

    Usage:
        from myNeRF_metrics import plot_lr_momentum_from_checkpoints
        plot_lr_momentum_from_checkpoints('results/hyperparam_grid')
    """
    path = os.path.join(results_dir, 'psnr_grid.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found - run run_lr_momentum_grid() first')
    data = np.load(path)
    plot_3d_surface(list(data['lr_values']), list(data['momentum_values']),
                    data['psnr_grid'], title='PSNR: Learning Rate x Momentum')

def run_iteration_experiment(NeRFClass, training_dir: str,
                              width=500, height=500,
                              num_stages=4, learning_rate=0.2,
                              num_views=36,
                              iteration_counts=None,
                              results_dir='results/iterations'):
    os.makedirs(results_dir, exist_ok=True)
    if iteration_counts is None:
        iteration_counts = [10, 25, 50, 100]

    psnr_vals, ssim_vals = [], []
    for n_iter in iteration_counts:
        print(f'\n{"="*50}\n  {n_iter} iterations/stage\n{"="*50}')
        nerf = NeRFClass(training_dir, width=width, height=height,
                         num_stages=num_stages,
                         num_iterations_per_stage=n_iter,
                         learning_rate=learning_rate)
        nerf.CHECKPOINT_DIR = os.path.join(results_dir, f'checkpoints_{n_iter}iter')
        os.makedirs(nerf.CHECKPOINT_DIR, exist_ok=True)
        nerf.prep()

        indices = np.round(np.linspace(0, len(nerf.ref_images_np) - 1, num_views)).astype(int)
        nerf.ref_images_np = [nerf.ref_images_np[i] for i in indices]
        nerf.sensors       = [nerf.sensors[i]       for i in indices]
        nerf.sensor_count  = num_views

        nerf.train()

        rendered = _render_views(nerf)
        metrics  = evaluate_all_views(rendered, nerf.ref_images_np)
        psnr_vals.append(metrics['mean_psnr'])
        ssim_vals.append(metrics['mean_ssim'])

        np.savez(os.path.join(results_dir, f'summary_{n_iter}iter.npz'),
                 n_iter=n_iter, losses=nerf.losses,
                 mean_psnr=metrics['mean_psnr'],
                 mean_ssim=metrics['mean_ssim'])
        print(f'\n  {n_iter} iter  PSNR={metrics["mean_psnr"]:.2f} dB'
              f'  SSIM={metrics["mean_ssim"]:.4f}')
        del nerf, rendered; gc.collect()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(iteration_counts, psnr_vals, 'o-', color='steelblue', linewidth=2)
    ax1.set_title('PSNR vs Iterations per Stage')
    ax1.set_xlabel('Iterations per stage')
    ax1.set_ylabel('Mean PSNR (dB)')
    ax1.grid(True, alpha=0.3)
    ax2.plot(iteration_counts, ssim_vals, 'o-', color='darkorange', linewidth=2)
    ax2.set_title('SSIM vs Iterations per Stage')
    ax2.set_xlabel('Iterations per stage')
    ax2.set_ylabel('Mean SSIM')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_vs_iterations.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_iteration_experiment_from_checkpoints(results_dir='results/iterations',
                                               iteration_counts=None):
    if iteration_counts is None:
        iteration_counts = [10, 25, 50, 100]
    psnr_vals, ssim_vals, found = [], [], []
    for n in iteration_counts:
        path = os.path.join(results_dir, f'summary_{n}iter.npz')
        if not os.path.exists(path):
            print(f'  [skip] {path} not found')
            continue
        data = np.load(path)
        psnr_vals.append(float(data['mean_psnr']))
        ssim_vals.append(float(data['mean_ssim']))
        found.append(n)
        print(f'  {n} iter  PSNR={float(data["mean_psnr"]):.2f} dB'
              f'  SSIM={float(data["mean_ssim"]):.4f}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(found, psnr_vals, 'o-', color='steelblue', linewidth=2)
    ax1.set_title('PSNR vs Iterations per Stage')
    ax1.set_xlabel('Iterations per stage')
    ax1.set_ylabel('Mean PSNR (dB)')
    ax1.grid(True, alpha=0.3)
    ax2.plot(found, ssim_vals, 'o-', color='darkorange', linewidth=2)
    ax2.set_title('SSIM vs Iterations per Stage')
    ax2.set_xlabel('Iterations per stage')
    ax2.set_ylabel('Mean SSIM')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
