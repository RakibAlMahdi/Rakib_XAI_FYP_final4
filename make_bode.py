# make_bode.py --------------------------------------------------
import numpy as np, scipy.signal as sig, matplotlib.pyplot as plt, pathlib
plt.rcParams.update({'font.size': 9})

FS = 1_000
specs = {'HP': dict(type='highpass',  f_c=25,  colour='C0'),
         'LP': dict(type='lowpass',   f_c=400, colour='C1')}

for name, cfg in specs.items():
    b, a = sig.butter(2, cfg['f_c'], btype=cfg['type'], fs=FS)
    w, h = sig.freqz(b, a, worN=2048, fs=FS)
    mag = 20*np.log10(np.abs(h))
    phase = np.unwrap(np.angle(h))*180/np.pi

    for arr, ylabel, suffix in [(mag, 'Magnitude (dB)', 'mag'),
                                (phase, 'Phase (deg)', 'phase')]:
        plt.figure(figsize=(3.2,2.2))
        plt.semilogx(w, arr, color=cfg['colour'])
        plt.title(f'{name} Butterworth ({cfg["f_c"]} Hz, N=2)')
        plt.xlabel('Frequency (Hz)'); plt.ylabel(ylabel); plt.grid(True, which='both')
        out = pathlib.Path(f'{name.lower()}_{suffix}.png')
        plt.tight_layout(); plt.savefig(out, dpi=300); plt.savefig(out.with_suffix('.pdf'))
        print('Wrote', out)