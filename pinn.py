import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling
import scipy.io
import os
from datetime import datetime

# Create output directory for saving figures
output_dir = 'pinn_output2'
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n{'='*70}")
print(f"PINN Schrödinger Equation Solver")
print(f"Output directory: {output_dir}")
print(f"Timestamp: {timestamp}")
print(f"{'='*70}\n")

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS device")
else:
    print("MPS device not found.")


print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU. Training will be slower.")
    
print(f"\n{'='*70}\n")

# Load data
print("Loading data from NLS.mat...")
data = scipy.io.loadmat('./NLS (1).mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
noise = 0.0

# Domain bounds
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])
N0 = 50
N_b = 50
N_f = 20000
layers = [2, 100, 100, 100, 100, 2]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

print(f"Data loaded successfully!")
print(f"  - Spatial domain: x ∈ [{x.min():.2f}, {x.max():.2f}] with {x.shape[0]} points")
print(f"  - Temporal domain: t ∈ [{t.min():.4f}, {t.max():.4f}] with {t.shape[0]} points")
print(f"  - Total data points: {x.shape[0] * t.shape[0]}")
print(f"\nPINN Configuration:")
print(f"  - Initial condition points (N0): {N0}")
print(f"  - Boundary points (N_b): {N_b}")
print(f"  - PDE collocation points (N_f): {N_f}")
print(f"  - Network architecture: {layers}")
print(f"\n{'='*70}\n")

def plot3D(x, t, y, filename='plot3d.png'):
    x_plot = x.squeeze(1)
    t_plot = t.squeeze(1)
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    F_xt = y
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.savefig(os.path.join(output_dir, filename.replace('.png', '_3d.png')), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename.replace('.png', '_3d.png')}")

def plot3D_Matrix(x, t, y, filename='plot3d_matrix.png'):
    X, T = x, t
    F_xt = y
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, F_xt, cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.savefig(os.path.join(output_dir, filename.replace('.png', '_3d.png')), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename.replace('.png', '_3d.png')}")

# Gradient norm plotting function
def moving_average(data, window_size=50):
    """Smooths data using a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def plot_gradient_norms(model, window=50):
    # Note: moving average shortens the array by (window-1)
    smooth_ic = moving_average(model.history_grad0, window)
    smooth_bc = moving_average(model.history_gradb, window)
    smooth_pde = moving_average(model.history_gradf, window)
    
    epochs = np.arange(len(smooth_ic))
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Gradient Norm Analysis (Smoothed w/ window={window})', fontsize=16)

    # --- Plot 1: All Gradients Combined ---
    axs[0, 0].plot(epochs, smooth_ic, label='Initial Condition', alpha=0.8)
    axs[0, 0].plot(epochs, smooth_bc, label='Boundary Condition', alpha=0.8)
    axs[0, 0].plot(epochs, smooth_pde, label='PDE Residual', alpha=0.8)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Combined Gradients (Comparison)')
    axs[0, 0].legend()

    # --- Plot 2: Initial Condition Only ---
    axs[0, 1].plot(epochs, smooth_ic, color='blue')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Initial Condition Gradient Norm')
    axs[0, 1].set_ylabel('L2 Norm')

    # --- Plot 3: Boundary Condition Only ---
    axs[1, 0].plot(epochs, smooth_bc, color='orange')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Boundary Condition Gradient Norm')
    axs[1, 0].set_ylabel('L2 Norm')
    axs[1, 0].set_xlabel('Epochs')

    # --- Plot 4: PDE Residual Only ---
    axs[1, 1].plot(epochs, smooth_pde, color='green')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('PDE Residual Gradient Norm')
    axs[1, 1].set_ylabel('L2 Norm')
    axs[1, 1].set_xlabel('Epochs')

    for ax in axs.flat:
        ax.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Sample initial and boundary points
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x, :]
u0 = Exact_u[idx_x, 0:1]
v0 = Exact_v[idx_x, 0:1]

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t, :]

X_f = lb + (ub - lb) * lhs(2, N_f)  # points for PDE Loss


class PINN(nn.Module):
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        super().__init__()
        'activation function'
        self.activation = nn.Tanh()
        'data prep'
        X0 = np.concatenate((x0, 0*x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # (ub[0], tb)
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.X0 = torch.tensor(X0, requires_grad=True).float().to(device)
        self.x0 = self.X0[:, 0:1]
        self.t0 = self.X0[:, 1:2]

        self.X_lb = torch.tensor(X_lb, requires_grad=True).float().to(device)
        self.x_lb = self.X_lb[:, 0:1]
        self.t_lb = self.X_lb[:, 1:2]

        self.X_ub = torch.tensor(X_ub, requires_grad=True).float().to(device)
        self.x_ub = self.X_ub[:, 0:1]
        self.t_ub = self.X_ub[:, 1:2]

        self.X_f = torch.tensor(X_f, requires_grad=True).float().to(device)
        self.x_f = self.X_f[:, 0:1]
        self.t_f = self.X_f[:, 1:2]

        self.u0 = torch.tensor(u0).float().to(device)
        self.v0 = torch.tensor(v0).float().to(device)

        #grad_history
        self.history_grad0 = []
        self.history_gradb = []
        self.history_gradf = []

        #loss weighting factors
        self.lambda_bc = 1.0
        self.lambda_pde = 1.0
        self.alpha = 1.0 #moving average factor

        # Neural Network layers
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0  # For the Optimizer
        self.loss_history = []
        
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def net_uv(self, X):
        uv = self.forward(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def net_uv_deriv(self, X):
        g = X.clone()
        g.requires_grad_(True)
        uv = self.forward(g)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        u_x_t = autograd.grad(u, g, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        v_x_t = autograd.grad(v, g, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        u_x = u_x_t[:, 0:1]
        v_x = v_x_t[:, 0:1]
        u_t = u_x_t[:, 1:2]
        v_t = v_x_t[:, 1:2]
        u_xx = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        v_xx = autograd.grad(v_x, g, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        return u_x, v_x, u_xx, v_xx, u_t, v_t

    def loss_func(self):
        # Initial condition loss
        u0_pred, v0_pred = self.net_uv(self.X0)
        Loss_IC = nn.MSELoss()(self.u0, u0_pred) + nn.MSELoss()(self.v0, v0_pred)
        
        # Boundary condition loss (periodic)
        u_lb_pred, v_lb_pred = self.net_uv(self.X_lb)
        u_ub_pred, v_ub_pred = self.net_uv(self.X_ub)
        u_x_ub_pred, v_x_ub_pred, _, _, _, _ = self.net_uv_deriv(self.X_ub)
        u_x_lb_pred, v_x_lb_pred, _, _, _, _ = self.net_uv_deriv(self.X_lb)
        Loss_BC = nn.MSELoss()(u_lb_pred, u_ub_pred) + nn.MSELoss()(v_lb_pred, v_ub_pred) + \
                  nn.MSELoss()(u_x_lb_pred, u_x_ub_pred) + nn.MSELoss()(v_x_lb_pred, v_x_ub_pred)
        
        # PDE residual loss
        u, v = self.net_uv(self.X_f)
        _, _, u_xx_pred, v_xx_pred, u_t_pred, v_t_pred = self.net_uv_deriv(self.X_f)
        f_u_pred = u_t_pred + 0.5*v_xx_pred + (u**2 + v**2)*v
        f_v_pred = v_t_pred - 0.5*u_xx_pred - (u**2 + v**2)*u
        Loss_PDE = nn.MSELoss()(f_u_pred, torch.zeros_like(f_u_pred)) + \
                   nn.MSELoss()(f_v_pred, torch.zeros_like(f_v_pred))
        
        
        return Loss_IC, Loss_BC, Loss_PDE

    def closure(self):
        optimizer.zero_grad()
        Loss_IC,Loss_BC,Loss_PDE = self.loss_func()

        #  Track Gradient Norms (Diagnostic)
        # We compute grads for each loss component independently
        grad_ic = autograd.grad(Loss_IC, self.parameters(), retain_graph=True)
        grad_bc = autograd.grad(Loss_BC, self.parameters(), retain_graph=True)
        grad_pde = autograd.grad(Loss_PDE, self.parameters(), retain_graph=True)

        #  calculate weights for losses
        with torch.no_grad(): # Ensure these updates don't track gradients
            max_grad_ic = torch.max(torch.stack([torch.max(torch.abs(g)) for g in grad_ic if g is not None]))
            mean_grad_pde = torch.mean(torch.stack([torch.mean(torch.abs(g)) for g in grad_pde if g is not None]))
            mean_grad_bc = torch.mean(torch.stack([torch.mean(torch.abs(g)) for g in grad_bc if g is not None]))

            lambda_hat_bc = max_grad_ic / (mean_grad_bc + 1e-8)
            lambda_hat_pde = max_grad_ic / (mean_grad_pde + 1e-8)

            # Exponential Moving Average update
            # .detach() ensures these are treated as constants in the final loss
            self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * lambda_hat_bc.detach()
            self.lambda_pde = self.alpha * self.lambda_pde + (1 - self.alpha) * lambda_hat_pde.detach()

        #Helper function to compute L2 norm of gradients
        def grad_norm(grads):
            valid_grads = [g.reshape(-1) for g in grads if g is not None]
            if not valid_grads: return 0.0
            return torch.norm(torch.cat(valid_grads)).item()
        self.history_grad0.append(grad_norm(grad_ic))
        self.history_gradb.append(grad_norm(grad_bc))
        self.history_gradf.append(grad_norm(grad_pde))

        #final loss and backprop
        loss = Loss_IC + (self.lambda_bc*Loss_BC) + (self.lambda_pde*Loss_PDE)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(f'Iter {self.iter}, Loss: {loss.item():.6e}')
        self.loss_history.append(loss.item())
        return loss

    def predict(self, X):
        x = torch.tensor(X, requires_grad=True).float().to(device)
        self.eval()
        u, v = self.net_uv(x)
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        return u, v


# ============================================================================
# TRAIN MODEL 1: ADAM OPTIMIZER
# ============================================================================
print("="*70)
print("TRAINING MODEL 1: ADAM OPTIMIZER")
print("="*70)

# Reset seed for fair comparison
torch.manual_seed(1234)
np.random.seed(1234)

model_adam = PINN(x0, u0, v0, tb, X_f, layers, lb, ub)
model_adam.to(device)

optimizer = optim.Adam(model_adam.parameters(), lr=0.001)
start_time = time.time()

for epoch in range(10000):
    loss = model_adam.closure()
    optimizer.step()

elapsed_adam = time.time() - start_time
print(f'\nAdam Training time: {elapsed_adam:.2f}s')

# ============================================================================
# TRAIN MODEL 2: L-BFGS OPTIMIZER
# ============================================================================
print("\n" + "="*70)
print("TRAINING MODEL 2: L-BFGS OPTIMIZER")
print("="*70)

# Reset seed for fair comparison
torch.manual_seed(1234)
np.random.seed(1234)

model_lbfgs = PINN(x0, u0, v0, tb, X_f, layers, lb, ub)
model_lbfgs.to(device)

optimizer = optim.LBFGS(model_lbfgs.parameters(), 
                        lr=1.0, 
                        max_iter=50000, 
                        max_eval=50000, 
                        history_size=50,
                        tolerance_grad=1e-5, 
                        tolerance_change=1.0 * np.finfo(float).eps,
                        line_search_fn="strong_wolfe")

start_time = time.time()
optimizer.step(model_lbfgs.closure)
elapsed_lbfgs = time.time() - start_time
print(f'\nL-BFGS Training time: {elapsed_lbfgs:.2f}s')

# ============================================================================
# COMPARE LOSS HISTORIES
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].semilogy(model_adam.loss_history, label='Adam', linewidth=2)
axes[0].semilogy(model_lbfgs.loss_history, label='L-BFGS', linewidth=2)
axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Zoomed in view of final losses
min_len = min(len(model_adam.loss_history), len(model_lbfgs.loss_history))
axes[1].semilogy(model_adam.loss_history[-min_len:], label='Adam', linewidth=2)
axes[1].semilogy(model_lbfgs.loss_history[-min_len:], label='L-BFGS', linewidth=2)
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Loss Comparison (Last Iterations)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'loss_comparison_{timestamp}.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: loss_comparison_{timestamp}.png")

# ============================================================================
# GENERATE PREDICTIONS FOR BOTH MODELS
# ============================================================================
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Adam predictions
u_pred_adam, v_pred_adam = model_adam.predict(X_star)
h_pred_adam = np.sqrt(u_pred_adam**2 + v_pred_adam**2)
U_pred_adam = u_pred_adam.reshape(X.shape)
V_pred_adam = v_pred_adam.reshape(X.shape)
H_pred_adam = h_pred_adam.reshape(X.shape)

# L-BFGS predictions
u_pred_lbfgs, v_pred_lbfgs = model_lbfgs.predict(X_star)
h_pred_lbfgs = np.sqrt(u_pred_lbfgs**2 + v_pred_lbfgs**2)
U_pred_lbfgs = u_pred_lbfgs.reshape(X.shape)
V_pred_lbfgs = v_pred_lbfgs.reshape(X.shape)
H_pred_lbfgs = h_pred_lbfgs.reshape(X.shape)

Exact_u_plot = Exact_u.T
Exact_v_plot = Exact_v.T
Exact_h_plot = Exact_h.T

# ============================================================================
# CALCULATE ERRORS
# ============================================================================
error_u_adam = np.linalg.norm(Exact_u_plot - U_pred_adam, 2) / np.linalg.norm(Exact_u_plot, 2)
error_v_adam = np.linalg.norm(Exact_v_plot - V_pred_adam, 2) / np.linalg.norm(Exact_v_plot, 2)
error_h_adam = np.linalg.norm(Exact_h_plot - H_pred_adam, 2) / np.linalg.norm(Exact_h_plot, 2)

error_u_lbfgs = np.linalg.norm(Exact_u_plot - U_pred_lbfgs, 2) / np.linalg.norm(Exact_u_plot, 2)
error_v_lbfgs = np.linalg.norm(Exact_v_plot - V_pred_lbfgs, 2) / np.linalg.norm(Exact_v_plot, 2)
error_h_lbfgs = np.linalg.norm(Exact_h_plot - H_pred_lbfgs, 2) / np.linalg.norm(Exact_h_plot, 2)

print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"\n{'Metric':<20} {'Adam':<20} {'L-BFGS':<20}")
print("-"*70)
print(f"{'Training Time (s)':<20} {elapsed_adam:<20.2f} {elapsed_lbfgs:<20.2f}")
print(f"{'Final Loss':<20} {model_adam.loss_history[-1]:<20.6e} {model_lbfgs.loss_history[-1]:<20.6e}")
print(f"{'L2 Error (u)':<20} {error_u_adam:<20.6e} {error_u_lbfgs:<20.6e}")
print(f"{'L2 Error (v)':<20} {error_v_adam:<20.6e} {error_v_lbfgs:<20.6e}")
print(f"{'L2 Error (|h|)':<20} {error_h_adam:<20.6e} {error_h_lbfgs:<20.6e}")
print("-"*70)

winner_loss = "Adam" if model_adam.loss_history[-1] < model_lbfgs.loss_history[-1] else "L-BFGS"
winner_accuracy = "Adam" if error_h_adam < error_h_lbfgs else "L-BFGS"
print(f"\n Lower Final Loss: {winner_loss}")
print(f" Better Accuracy: {winner_accuracy}")

# ============================================================================
# SIDE-BY-SIDE VISUALIZATION
# ============================================================================
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 4, figure=fig)

# Row 1: |h| predictions
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.contourf(T, X, H_pred_adam, 20, cmap='rainbow')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title('|h(x,t)| - Adam', fontweight='bold')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.contourf(T, X, H_pred_lbfgs, 20, cmap='rainbow')
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('|h(x,t)| - L-BFGS', fontweight='bold')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.contourf(T, X, Exact_h_plot, 20, cmap='rainbow')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_title('|h(x,t)| - Exact', fontweight='bold')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

ax4 = fig.add_subplot(gs[0, 3])
error_diff = np.abs(H_pred_adam - Exact_h_plot) - np.abs(H_pred_lbfgs - Exact_h_plot)
im4 = ax4.contourf(T, X, error_diff, 20, cmap='RdBu_r')
ax4.set_xlabel('t')
ax4.set_ylabel('x')
ax4.set_title('Error Diff (Adam-LBFGS)', fontweight='bold')
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im4, cax=cax)

# Row 2: u predictions
ax5 = fig.add_subplot(gs[1, 0])
im5 = ax5.contourf(T, X, U_pred_adam, 20, cmap='rainbow')
ax5.set_xlabel('t')
ax5.set_ylabel('x')
ax5.set_title('u(x,t) - Adam', fontweight='bold')
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im5, cax=cax)

ax6 = fig.add_subplot(gs[1, 1])
im6 = ax6.contourf(T, X, U_pred_lbfgs, 20, cmap='rainbow')
ax6.set_xlabel('t')
ax6.set_ylabel('x')
ax6.set_title('u(x,t) - L-BFGS', fontweight='bold')
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im6, cax=cax)

ax7 = fig.add_subplot(gs[1, 2])
im7 = ax7.contourf(T, X, Exact_u_plot, 20, cmap='rainbow')
ax7.set_xlabel('t')
ax7.set_ylabel('x')
ax7.set_title('u(x,t) - Exact', fontweight='bold')
divider = make_axes_locatable(ax7)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im7, cax=cax)

ax8 = fig.add_subplot(gs[1, 3])
im8 = ax8.contourf(T, X, np.abs(U_pred_adam - Exact_u_plot), 20, cmap='Reds')
ax8.set_xlabel('t')
ax8.set_ylabel('x')
ax8.set_title('|Error u| - Adam', fontweight='bold')
divider = make_axes_locatable(ax8)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im8, cax=cax)

# Row 3: v predictions
ax9 = fig.add_subplot(gs[2, 0])
im9 = ax9.contourf(T, X, V_pred_adam, 20, cmap='rainbow')
ax9.set_xlabel('t')
ax9.set_ylabel('x')
ax9.set_title('v(x,t) - Adam', fontweight='bold')
divider = make_axes_locatable(ax9)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im9, cax=cax)

ax10 = fig.add_subplot(gs[2, 1])
im10 = ax10.contourf(T, X, V_pred_lbfgs, 20, cmap='rainbow')
ax10.set_xlabel('t')
ax10.set_ylabel('x')
ax10.set_title('v(x,t) - L-BFGS', fontweight='bold')
divider = make_axes_locatable(ax10)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im10, cax=cax)

ax11 = fig.add_subplot(gs[2, 2])
im11 = ax11.contourf(T, X, Exact_v_plot, 20, cmap='rainbow')
ax11.set_xlabel('t')
ax11.set_ylabel('x')
ax11.set_title('v(x,t) - Exact', fontweight='bold')
divider = make_axes_locatable(ax11)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im11, cax=cax)

ax12 = fig.add_subplot(gs[2, 3])
im12 = ax12.contourf(T, X, np.abs(V_pred_lbfgs - Exact_v_plot), 20, cmap='Reds')
ax12.set_xlabel('t')
ax12.set_ylabel('x')
ax12.set_title('|Error v| - L-BFGS', fontweight='bold')
divider = make_axes_locatable(ax12)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im12, cax=cax)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'full_comparison_{timestamp}.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: full_comparison_{timestamp}.png")

# ============================================================================
# 3D COMPARISON
# ============================================================================
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(T, X, H_pred_adam, cmap='rainbow')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_zlabel('|h(x,t)|')
ax1.set_title(f'Adam (Error: {error_h_adam:.4e})', fontweight='bold')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(T, X, H_pred_lbfgs, cmap='rainbow')
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_zlabel('|h(x,t)|')
ax2.set_title(f'L-BFGS (Error: {error_h_lbfgs:.4e})', fontweight='bold')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(T, X, Exact_h_plot, cmap='rainbow')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_zlabel('|h(x,t)|')
ax3.set_title('Exact Solution', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'3d_comparison_{timestamp}.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 3d_comparison_{timestamp}.png")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print(f"All figures saved to: {output_dir}/")
print("="*70)
# ===========================================================================
# PLOT GRADIENT NORMS
# ===========================================================================
plot_gradient_norms(model_adam)
plot_gradient_norms(model_lbfgs)
plt.savefig(os.path.join(output_dir, f'gradient_norms_{timestamp}.png'), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 3d_comparison_{timestamp}.png")
