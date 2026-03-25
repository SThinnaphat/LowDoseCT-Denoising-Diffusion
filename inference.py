"""
inference.py
------------
Standalone inference script for FA-CFG-LDCT.
Runs denoising on a single DICOM file or a folder of DICOM files.

Usage:
    # Single file
    python inference.py --input scan.IMA --output denoised.png --config B --w 1.5

    # Folder of DICOM files
    python inference.py --input ./dicom_folder/ --output ./output/ --config B --w 1.5

    # Use FANP model with standard sampling (no CFG)
    python inference.py --input scan.IMA --output denoised.png --config C
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
import matplotlib.pyplot as plt


# ── Model definitions (copy from notebook) ────────────────────────────────
class SinEmb(nn.Module):
    def __init__(self, d): super().__init__(); self.d = d
    def forward(self, t):
        h = self.d // 2; e = math.log(10000) / (h - 1)
        e = torch.exp(torch.arange(h, device=t.device) * -e)
        e = t[:,None].float() * e[None,:]
        return torch.cat([e.sin(), e.cos()], -1)

class TMLP(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        self.m = nn.Sequential(SinEmb(d), nn.Linear(d,o), nn.SiLU(), nn.Linear(o,o))
    def forward(self, t): return self.m(t)

class RB(nn.Module):
    def __init__(self, i, o, td, g=8):
        super().__init__()
        self.n1=nn.GroupNorm(g,i); self.c1=nn.Conv2d(i,o,3,1,1)
        self.n2=nn.GroupNorm(g,o); self.c2=nn.Conv2d(o,o,3,1,1)
        self.tp=nn.Linear(td,o)
        self.sk=nn.Conv2d(i,o,1) if i!=o else nn.Identity()
        self.a=nn.SiLU()
    def forward(self, x, t):
        h = self.c1(self.a(self.n1(x)))
        h = h + self.tp(self.a(t))[:,:,None,None]
        return self.c2(self.a(self.n2(h))) + self.sk(x)

class SA(nn.Module):
    def __init__(self, c, h=4):
        super().__init__()
        self.h=h; self.n=nn.GroupNorm(8,c)
        self.qkv=nn.Conv2d(c,c*3,1); self.p=nn.Conv2d(c,c,1)
        self.sc=(c//h)**-.5
    def forward(self, x):
        B,C,H,W = x.shape
        qkv = self.qkv(self.n(x)).reshape(B,3,self.h,C//self.h,H*W)
        q,k,v = qkv[:,0],qkv[:,1],qkv[:,2]
        a = (torch.einsum('bhci,bhcj->bhij',q,k)*self.sc).softmax(-1)
        return x + self.p(torch.einsum('bhij,bhcj->bhci',a,v).reshape(B,C,H,W))

class Dn(nn.Module):
    def __init__(self, c): super().__init__(); self.c=nn.Conv2d(c,c,3,2,1)
    def forward(self, x): return self.c(x)

class Up_(nn.Module):
    def __init__(self, c): super().__init__(); self.c=nn.Conv2d(c,c,3,1,1)
    def forward(self, x): return self.c(F.interpolate(x,scale_factor=2,mode='nearest'))

class CUNet(nn.Module):
    def __init__(self, ic=2, oc=1, b=64, td=256):
        super().__init__()
        self.te=TMLP(td,td); self.inc=nn.Conv2d(ic,b,3,1,1)
        self.d1a=RB(64,64,td);   self.d1b=RB(64,64,td);   self.p1=Dn(64)
        self.d2a=RB(64,128,td);  self.d2b=RB(128,128,td); self.p2=Dn(128)
        self.d3a=RB(128,256,td); self.d3b=RB(256,256,td); self.p3=Dn(256)
        self.ba=RB(256,512,td);  self.at=SA(512);          self.bb=RB(512,512,td)
        self.u3=Up_(512); self.u3a=RB(768,256,td); self.u3b=RB(256,256,td)
        self.u2=Up_(256); self.u2a=RB(384,128,td); self.u2b=RB(128,128,td)
        self.u1=Up_(128); self.u1a=RB(192,64,td);  self.u1b=RB(64,64,td)
        self.out=nn.Sequential(nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,oc,3,1,1))
    def forward(self, xn, t, c):
        te = self.te(t)
        x1=self.d1b(self.d1a(self.inc(torch.cat([xn,c],1)),te),te)
        x2=self.d2b(self.d2a(self.p1(x1),te),te)
        x3=self.d3b(self.d3a(self.p2(x2),te),te)
        x4=self.bb(self.at(self.ba(self.p3(x3),te)),te)
        x=self.u3b(self.u3a(torch.cat([self.u3(x4),x3],1),te),te)
        x=self.u2b(self.u2a(torch.cat([self.u2(x), x2],1),te),te)
        x=self.u1b(self.u1a(torch.cat([self.u1(x), x1],1),te),te)
        return self.out(x)

def cos_sched(T, s=.008):
    t = torch.linspace(0, T, T+1) / T
    ac = torch.cos((t+s)/(1+s)*math.pi/2)**2
    ac = ac / ac[0]
    return torch.clamp(1-ac[1:]/ac[:-1], .0001, .9999)

class Diff(nn.Module):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model; self.T = T
        b = cos_sched(T); a = 1-b
        ac = torch.cumprod(a, 0)
        acp = torch.cat([torch.tensor([1.]), ac[:-1]])
        for n,v in [('b',b),('a',a),('ac',ac),('sac',ac.sqrt()),
                    ('s1m',(1-ac).sqrt()),('sra',(1/a).sqrt()),
                    ('pv', b*(1-acp)/(1-ac))]:
            self.register_buffer(n, v)

    @torch.no_grad()
    def sample_partial(self, cond, start_step=100):
        B,_,H,W = cond.shape
        t_start = torch.full((B,), start_step, device=cond.device, dtype=torch.long)
        noise = torch.randn_like(cond)
        x = self.sac[t_start][:,None,None,None]*cond + \
            self.s1m[t_start][:,None,None,None]*noise
        steps = list(range(start_step, -1, -(max(start_step//50, 1))))
        for i in range(len(steps)-1):
            t, tp = steps[i], steps[i+1]
            tb = torch.full((B,), t, device=cond.device, dtype=torch.long)
            eps = self.model(x, tb, cond)
            at = self.ac[t]
            ap = self.ac[tp] if tp > 0 else torch.tensor(1., device=cond.device)
            x0_ = ((x-(1-at).sqrt()*eps)/at.sqrt()).clamp(-1,2)
            x = ap.sqrt()*x0_ + (1-ap).sqrt()*eps
        return x.clamp(0,1)

    @torch.no_grad()
    def ddim_cfg(self, c, steps=50, w=1.5):
        B,_,H,W = c.shape
        ss = self.T // steps
        ts = list(reversed(range(0, self.T, ss)))
        x = torch.randn(B, 1, H, W, device=c.device)
        null = torch.zeros_like(c)
        for i in range(len(ts)):
            t, tp = ts[i], (ts[i+1] if i+1<len(ts) else 0)
            tb = torch.full((B,), t, device=c.device, dtype=torch.long)
            eps_c = self.model(x, tb, c)
            eps_u = self.model(x, tb, null)
            eps = eps_u + w*(eps_c - eps_u)
            at = self.ac[t]
            ap = self.ac[tp] if tp > 0 else torch.tensor(1., device=c.device)
            x0_ = ((x-(1-at).sqrt()*eps)/at.sqrt()).clamp(-1,2)
            x = ap.sqrt()*x0_ + (1-ap).sqrt()*eps
        return x.clamp(0,1)


# ── Preprocessing ──────────────────────────────────────────────────────────
def load_dicom(path, lo=-1000, hi=2000):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = img * float(getattr(ds,'RescaleSlope',1)) + float(getattr(ds,'RescaleIntercept',0))
    img = np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)
    return img

def to_tensor(img, device):
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='FA-CFG-LDCT inference')
    parser.add_argument('--input',  required=True,
                        help='Input DICOM file (.IMA/.dcm) or folder')
    parser.add_argument('--output', required=True,
                        help='Output PNG file or folder')
    parser.add_argument('--config', default='B', choices=['A','B','C','D'],
                        help='Model config to use (default: B)')
    parser.add_argument('--w',      type=float, default=1.5,
                        help='CFG guidance weight (only for configs B and D, default: 1.5)')
    parser.add_argument('--t_star', type=int,   default=100,
                        help='Warm-start timestep (default: 100)')
    parser.add_argument('--steps',  type=int,   default=50,
                        help='Number of DDIM steps (default: 50)')
    parser.add_argument('--ckpt_dir', default='./checkpoints',
                        help='Directory containing checkpoint files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    config_map = {'A':'config_A','B':'config_B','C':'config_C','D':'config_D'}
    ckpt_path = os.path.join(args.ckpt_dir, f'{config_map[args.config]}_best.pt')
    if not os.path.exists(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        print('Run: python download_weights.py')
        return

    print(f'Loading Config {args.config} from {ckpt_path}')
    model = Diff(CUNet(), T=1000).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['m'])
    model.eval()
    print(f'Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # Collect input files
    if os.path.isfile(args.input):
        files = [args.input]
        out_paths = [args.output]
    else:
        exts = ('.IMA', '.ima', '.dcm', '.DCM')
        files = [os.path.join(args.input, f)
                 for f in sorted(os.listdir(args.input))
                 if f.endswith(exts)]
        os.makedirs(args.output, exist_ok=True)
        out_paths = [os.path.join(args.output,
                     os.path.splitext(os.path.basename(f))[0] + '_denoised.png')
                     for f in files]
        print(f'Found {len(files)} DICOM files in {args.input}')

    use_cfg = args.config in ('B', 'D')

    for fpath, opath in zip(files, out_paths):
        print(f'  Processing {os.path.basename(fpath)} ...', end=' ', flush=True)
        img = load_dicom(fpath)
        x = to_tensor(img, device)

        if use_cfg:
            out = model.ddim_cfg(x, steps=args.steps, w=args.w)
        else:
            out = model.sample_partial(x, start_step=args.t_star)

        out_np = out[0,0].cpu().numpy()

        # Save side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img,    cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Low-Dose Input'); axes[0].axis('off')
        axes[1].imshow(out_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Denoised (Config {args.config}'
                          + (f', w={args.w}' if use_cfg else '') + ')')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(opath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'saved to {opath}')

    print('\nDone.')


if __name__ == '__main__':
    main()
