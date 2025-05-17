import argparse, numpy as np, torch, pandas as pd
from sklearn.metrics import roc_curve
from datasets import SegmentPCGDataset
from torch.utils.data import DataLoader
from inference import load_model

parser = argparse.ArgumentParser(description="Find clinical threshold with target sens & spec")
parser.add_argument('--meta_csv', required=True)
parser.add_argument('--weights', default='best_combined.pth')
parser.add_argument('--target_sens', type=float, default=0.87)
parser.add_argument('--target_spec', type=float, default=0.80)
parser.add_argument('--out_ckpt', default=None, help='If given, updates checkpoint with thr_clin')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _ = load_model(args.weights, device)

# build DataLoader (no augment)
df = pd.read_csv(args.meta_csv)
val_ds = SegmentPCGDataset(df, augment=False)
loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2,
                    collate_fn=lambda b: (torch.stack([x for x,_,_ in b]),
                                           torch.tensor([y for _,y,_ in b]),
                                           [pid for *_, pid in b]))

# ----- patient-level aggregation -----
seg_scores, seg_labels, seg_pids = [], [], []
with torch.no_grad():
    for xb, yb, pids in loader:
        probs = torch.sigmoid(model(xb.to(device))).cpu().tolist()
        seg_scores.extend(probs); seg_labels.extend(yb.tolist()); seg_pids.extend(pids)

pat_dict = {}
lab_dict = {}
for pid, s, lab in zip(seg_pids, seg_scores, seg_labels):
    pat_dict.setdefault(pid, []).append(s)
    lab_dict[pid] = lab

scores = np.array([np.mean(v) for v in pat_dict.values()])
labels = np.array([lab_dict[pid] for pid in pat_dict.keys()])

fpr, tpr, thr = roc_curve(labels, scores)
spec = 1 - fpr

# distance in 2-D
cost = np.abs(tpr-args.target_sens)+np.abs(spec-args.target_spec)
idx  = cost.argmin()
thr_clin = float(thr[idx])
print(f'Clinical threshold = {thr_clin:.3f}  (sens {tpr[idx]:.3f}  spec {spec[idx]:.3f})')

if args.out_ckpt:
    ckpt = torch.load(args.weights, map_location='cpu')
    ckpt['thr_clin'] = thr_clin
    torch.save(ckpt, args.out_ckpt)
    print('Updated checkpoint with thr_clin') 