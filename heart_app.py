"""heart_app.py – KivyMD GUI for Heart-Sound XAI Demo

Run on desktop (Windows/macOS/Linux).  Mobile build: use buildozer with
pyaudio==0.2.13, matplotlib-base, kivy-garden.matplotlib and kivymd.

Features
---------
1. Record button – start capturing microphone, 1 kHz mono, 16-bit.
2. Stop button – stop & auto-save under ~/Documents/recording_<timestamp>.wav
3. Load button – file-chooser for *.wav
4. Plot area – matplotlib canvas; x axis = sample index, y axis = amplitude.
   Live updating during record & playback.
5. Play button – play currently loaded / recorded wav.
6. Explain button – after a wav is loaded, runs the PyTorch model
   best_combined.pth via xai_utils.generate_explanation and shows SHAP mask +
   attention on the same canvas.

NB: to keep the example compact many production concerns (threading, error
handling, mobile permissions) are simplified.
"""

from __future__ import annotations

import os, time, wave, threading, queue, datetime, sys
import numpy as np
import pyaudio
import torchaudio, torch
from kivy.config import Config
Config.set("graphics", "width", "480"); Config.set("graphics", "height", "800")
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.properties import BooleanProperty, StringProperty
from kivy.utils import platform
from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.dialog import MDDialog

import matplotlib
matplotlib.use("Agg")
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt

# ---------- ML imports ----------
sys.path.insert(0, os.path.join(os.getcwd(), "src"))  # adjust if needed
from xai_utils import load_model, prepare_explainer, generate_explanation, visualise_explanation

# --- locate checkpoint ----------------
POSSIBLE_MODEL = [
    os.path.join(os.getcwd(), "best_combined.pth"),
    os.path.join(os.getcwd(), "src", "best_combined.pth"),
    os.path.join(os.path.expanduser("~"), "best_combined.pth"),
]
MODEL_PATH = next((p for p in POSSIBLE_MODEL if os.path.exists(p)), None)

if MODEL_PATH is None:
    print("[WARN] best_combined.pth not found; Explain button will ask user to locate the file.")

# device for torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# constant sample rate
SR = 1000

# ---------- Audio controller (blocking I/O moved to thread) ----------
class AudioController:
    def __init__(self, update_cb):
        self.pa = pyaudio.PyAudio()
        self.fs = SR           # 1 kHz target
        self.chunk = 1024
        self.stream = None
        self.frames: list[bytes] = []
        self.is_recording = False
        self.update_cb = update_cb  # function(new_np_int16_array)
        self.play_wave: wave.Wave_read | None = None
        self.play_thread: threading.Thread | None = None

    # ---------------- record ----------------
    def start_record(self):
        if self.is_recording:
            return
        self.frames.clear()
        self.is_recording = True
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=self.fs,
                                   input=True, frames_per_buffer=self.chunk,
                                   stream_callback=self._rec_callback)
        self.stream.start_stream()

    def _rec_callback(self, in_data, frame_count, time_info, status):
        if not self.is_recording:
            return (None, pyaudio.paComplete)
        self.frames.append(in_data)
        self.update_cb(np.frombuffer(in_data, dtype=np.int16))
        return (None, pyaudio.paContinue)

    def stop_record(self) -> str | None:
        if not self.is_recording:
            return None
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream(); self.stream.close(); self.stream = None
        if not self.frames:
            return None
        # save wav
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.path.expanduser("~"), "Documents", f"recording_{ts}.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        wf = wave.open(out_path, "wb")
        wf.setnchannels(1); wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16)); wf.setframerate(self.fs)
        wf.writeframes(b"".join(self.frames)); wf.close()
        return out_path

    # ---------------- playback ----------------
    def play_wav(self, path: str):
        if self.play_thread and self.play_thread.is_alive():
            return  # already playing
        def _run():
            with wave.open(path, "rb") as wf:
                stream = self.pa.open(format=self.pa.get_format_from_width(wf.getsampwidth()),
                                       channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
                data = wf.readframes(self.chunk)
                while data:
                    stream.write(data)
                    self.update_cb(np.frombuffer(data, dtype=np.int16))
                    data = wf.readframes(self.chunk)
                stream.stop_stream(); stream.close()
        self.play_thread = threading.Thread(target=_run, daemon=True)
        self.play_thread.start()

# ---------------------------------------------------------------------
KV = """
BoxLayout:
    orientation: 'vertical'
    MDTopAppBar:
        title: 'Heart-Sound XAI'
        elevation: 5
    BoxLayout:
        id: plot_box
        size_hint_y: 0.55
    MDLabel:
        id: status_lbl
        text: 'Ready'
        halign: 'center'
        theme_text_color: 'Primary'
    MDLabel:
        id: time_lbl
        text: '0.0s'
        halign: 'center'
        theme_text_color: 'Secondary'
    BoxLayout:
        size_hint_y: 0.18
        spacing: dp(10)
        padding: dp(10)
        MDRaisedButton:
            text: 'Record'
            on_release: app.start_record()
        MDRaisedButton:
            text: 'Stop'
            on_release: app.stop_record()
        MDRaisedButton:
            text: 'Load'
            on_release: app.open_file_manager()
        MDRaisedButton:
            text: 'Play'
            on_release: app.play_audio()
        MDRaisedButton:
            text: 'Explain'
            on_release: app.explain_audio()
"""

# ---------------------------------------------------------------------
class HeartXAIApp(MDApp):
    recording = BooleanProperty(False)
    current_file: str | None = None

    def build(self):
        self.theme_cls.primary_palette = "Teal"
        self.controller = AudioController(self.on_new_audio_chunk)
        self.fig, self.ax = plt.subplots(figsize=(5,3))
        self.canvas = FigureCanvasKivyAgg(self.fig)
        self.wave_data = np.zeros(1, dtype=np.int16)
        root = Builder.load_string(KV)
        root.ids.plot_box.add_widget(self.canvas)
        self.file_manager = MDFileManager(select_path=self.select_file, exit_manager=lambda *_: None)
        return root

    # ---------------- live plot update ----------------
    @mainthread
    def on_new_audio_chunk(self, chunk: np.ndarray):
        self.wave_data = np.append(self.wave_data, chunk)
        if len(self.wave_data) > 10_000:
            self.wave_data = self.wave_data[-10_000:]
        self.ax.clear()
        t = np.arange(len(self.wave_data)) / SR
        self.ax.plot(t, self.wave_data, color='black', linewidth=0.6)
        self.ax.set_ylim(-32768, 32767)
        self.ax.set_xlabel("time (s)"); self.ax.set_ylabel("amplitude")
        self.canvas.draw()

    # ---------------- record/stop ----------------
    def start_record(self):
        if self.recording:
            return
        self.wave_data = np.zeros(1, dtype=np.int16)
        self.controller.start_record()
        self.root.ids.status_lbl.text = "Recording…"
        self.recording = True

    def stop_record(self):
        if not self.recording:
            return
        path = self.controller.stop_record()
        self.recording = False
        if path:
            self.current_file = path
            self.root.ids.status_lbl.text = f"Saved: {os.path.basename(path)}"
        else:
            self.root.ids.status_lbl.text = "Recording cancelled"

    # ---------------- load & play ----------------
    def open_file_manager(self):
        self.file_manager.show(os.path.expanduser("~"))
    def select_file(self, path):
        self.file_manager.close()
        if path and path.lower().endswith('.wav'):
            self.current_file = path
            Snackbar(text=f"Loaded {os.path.basename(path)}").open()
            # draw static waveform
            wav, sr = torchaudio.load(path); wav = wav.squeeze(0).numpy()
            self.ax.clear(); self.ax.plot(wav, color='black', linewidth=0.6)
            self.ax.set_xlabel("time (s)"); self.ax.set_ylabel("amplitude")
            self.canvas.draw()
        else:
            Snackbar(text="Please select a WAV file").open()

    def play_audio(self):
        if not self.current_file:
            Snackbar(text="No file loaded").open(); return
        self.controller.play_wav(self.current_file)
        self.root.ids.status_lbl.text = "Playing…"

    # ---------------- explain ----------------
    def explain_audio(self):
        if not self.current_file:
            Snackbar(text="Load a WAV first").open(); return
        self.root.ids.status_lbl.text = "Explaining…"
        # run in thread to avoid UI freeze
        threading.Thread(target=self._run_explain, daemon=True).start()

    def _run_explain(self):
        model = getattr(self, '_cached_model', None)
        explainer = getattr(self, '_cached_expl', None)
        if model is None:
            path = MODEL_PATH
            if path is None or not os.path.exists(path):
                # prompt user to pick checkpoint via file manager (blocking) on UI thread
                from kivy.app import App; ev = threading.Event()
                def ask_path():
                    fm = MDFileManager(select_path=lambda p: (App.get_running_app().__setattr__('_model_sel', p), fm.close(), ev.set()),
                                       exit_manager=lambda *_: ev.set())
                    fm.show(os.path.expanduser('~'))
                Clock.schedule_once(lambda dt: ask_path(), 0)
                ev.wait()
                path = getattr(self, '_model_sel', None)
                if path is None:
                    self.root.ids.status_lbl.text = "Checkpoint not provided."; return
            model = load_model(path, device);
            explainer = prepare_explainer(model, background_size=8, device=device)
            self._cached_model = model; self._cached_expl = explainer

        # -------- preprocessing same as training --------
        wav, sr = torchaudio.load(self.current_file)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        # band-pass 25-400 Hz
        wav = torchaudio.functional.highpass_biquad(wav, SR, 25)
        wav = torchaudio.functional.lowpass_biquad(wav, SR, 400)
        wav = wav.squeeze(0)
        if wav.numel() < 10_000:
            wav = torch.nn.functional.pad(wav, (0, 10_000 - wav.numel()))
        else:
            wav = wav[:10_000]

        info = generate_explanation(wav, model, explainer)
        @mainthread
        def _plot():
            # draw using helper which creates its own figure; grab it
            visualise_explanation(info, show_attention=True, save_path=None)
            new_fig = plt.gcf()
            self.canvas.figure = new_fig
            self.canvas.draw()
        _plot()
        self.root.ids.status_lbl.text = f"Done – Prob {info['prob']:.2f}"

    # ---------------- timer ----------------
    def on_start(self):
        self.seconds = 0.0
        Clock.schedule_interval(self._tick, 0.1)

    def _tick(self, dt):
        lbl = self.root.ids.get('time_lbl')
        if not lbl:
            return
        if self.recording or (self.controller.play_thread and self.controller.play_thread.is_alive()):
            self.seconds += dt
            lbl.text = f"{self.seconds:05.1f}s"
        else:
            self.seconds = 0.0
            lbl.text = "0.0s"

if __name__ == "__main__":
    HeartXAIApp().run() 