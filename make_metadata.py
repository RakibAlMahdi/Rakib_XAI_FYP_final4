import pandas as pd, os, argparse
from utils import load_metadata, load_physionet2016_metadata, wav_paths_for_patient

parser = argparse.ArgumentParser(description="Build combined metadata CSV for CirCor + PhysioNet16")
parser.add_argument('--circor_csv', required=True)
parser.add_argument('--circor_wav_dir', required=True)
parser.add_argument('--physio16_csv', required=True)
parser.add_argument('--physio16_wav_dir', required=True)
parser.add_argument('--out_csv', default='meta_all.csv')
args = parser.parse_args()

# CirCor
cir_md = load_metadata(args.circor_csv)
rows = []
for pid, lbl in zip(cir_md["Patient ID"], cir_md["label"]):
    for wp in wav_paths_for_patient(pid, args.circor_wav_dir):
        rows.append({'wav_path': wp, 'label': int(lbl), 'patient_id': str(pid)})

# Physio16
phys_md = load_physionet2016_metadata(args.physio16_csv)
for wav, lbl in phys_md.values:
    wp = os.path.join(args.physio16_wav_dir, wav)
    if os.path.exists(wp):
        rows.append({'wav_path': wp, 'label': int(lbl), 'patient_id': wav.split('.')[0]})

pd.DataFrame(rows).to_csv(args.out_csv, index=False)
print(f"Wrote {len(rows)} rows to {args.out_csv}") 