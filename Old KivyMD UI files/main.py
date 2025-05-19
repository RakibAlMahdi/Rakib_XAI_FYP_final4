from kivy.config import Config
Config.set('graphics', 'width', '480')
Config.set('graphics', 'height', '800')

import os
import time
import wave
import pyaudio
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for better performance
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import random
from kivy.utils import platform

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.audio import SoundLoader
try:
    from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
except ImportError:
    # Fallback if not available
    print("kivy_garden.matplotlib not available, using placeholder")
    from kivy.uix.widget import Widget
    class FigureCanvasKivyAgg(Widget):
        def __init__(self, figure=None, **kwargs):
            super(FigureCanvasKivyAgg, self).__init__(**kwargs)
            self.figure = figure

from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.properties import (
    StringProperty, NumericProperty, ObjectProperty, BooleanProperty
)

# Set color scheme
Window.clearcolor = (0.9, 0.9, 0.9, 1)

from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton
from kivymd.uix.filemanager import MDFileManager

from pydub import AudioSegment
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup

# Import heart sound analysis model
from heart_sound_model import HeartSoundClassifier

# Import auto-analysis functionality
from auto_analyze import start_background_analysis, show_pending_analysis_results

class AudioController:
    """Controller for audio recording and playback"""
    
    def __init__(self):
        """Initialize audio controller"""
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recorded_frames = []
        self.recording = False
        self.playing = False
        self.paused = False
        
        # Configure sampling rates
        self.device_sample_rate = 44100  # Default device recording rate (iPhone/standard audio)
        self.physionet_sample_rate = 2000  # PhysioNet heart sound dataset rate
        self.sample_rate = self.device_sample_rate  # Current active sample rate
        self.file_sample_rate = None  # Sample rate of loaded file
        
        self.channels = 1  # Mono
        self.chunk = 1024  # Number of frames per buffer
        self.format = pyaudio.paInt16  # 16-bit resolution
        self.waveform_data = np.array([], dtype=np.int16)  # For plotting
        self.device_info = self.pa.get_default_input_device_info()
        self.device_name = self.device_info.get('name', 'Unknown')
        
        # For time-based position estimation
        self.play_start_time = 0
        self.pause_time = 0
        self.pause_position = 0
        self.accumulated_pause_time = 0
        
        # Heart sound analysis
        self.heart_sound_classifier = HeartSoundClassifier()
        self.analysis_result = None
        self.current_file_path = None
        self.playback_sound = None  # To hold SoundLoader object
        
    def clear(self):
        """Clear current recording"""
        self.frames = []
        # Reset to empty array instead of zeros
        self.waveform_data = np.array([], dtype=np.int16)
        
    def start_recording(self):
        """Start recording audio"""
        # Initialize recording variables
        self.frames = []
        self.recorded_frames = []
        # Initialize with empty data
        self.waveform_data = np.array([], dtype=np.int16)
        self.recording = True
        
        # Open a stream to record audio
        self.stream = self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.recording_callback
        )
        self.stream.start_stream()
        
    def recording_callback(self, in_data, frame_count, time_info, status):
        """Callback function for recording audio
        
        Args:
            in_data (bytes): Input audio data
            frame_count (int): Number of frames
            time_info (dict): Timing information
            status (int): Status flag
            
        Returns:
            tuple: (None, flag) where flag is a pyaudio continue flag
        """
        # Convert the raw bytes to a numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Save the raw data for recording
        self.frames.append(in_data)
        
        # Save the processed data for visualization
        if hasattr(self, 'recorded_frames'):
            self.recorded_frames.append(audio_data.copy())
        
        # Update the waveform data for visualization
        self.update_waveform_data(audio_data)
        
        return (None, pyaudio.paContinue)
        
    def update_waveform_data(self, new_data):
        """Update the waveform data with new audio data
        
        Args:
            new_data (numpy.ndarray): New audio data
        """
        if not hasattr(self, 'waveform_data'):
            # Initialize waveform data if not exist
            self.waveform_data = np.array([], dtype=np.int16)
            # Initialize a timestamp when we start recording
            self.recording_start_time = time.time()
            
        # Ensure new_data is a numpy array and is not empty
        if not isinstance(new_data, np.ndarray):
            new_data = np.array(new_data, dtype=np.int16)
            
        if len(new_data) == 0:
            return
        
        # Update waveform data for visualization
        if self.recording:
            # For real-time waveform recording, we want to keep all the data
            # and plot it as a continuous stream from left to right
            
            # If this is our first recording chunk, set the start time
            if len(self.waveform_data) == 0 and not hasattr(self, 'recording_start_time'):
                self.recording_start_time = time.time()
                print(f"Starting recording at {self.recording_start_time}")
            
            # Always append new data to the end (left to right progression)
            self.waveform_data = np.append(self.waveform_data, new_data)
            
            # Calculate recording duration
            if hasattr(self, 'recording_start_time'):
                self.recording_duration = time.time() - self.recording_start_time
                if random.random() < 0.01:  # Only log occasionally
                    print(f"Recording duration: {self.recording_duration:.2f}s, samples: {len(self.waveform_data)}")
                    
            # Print debug info occasionally
            if random.random() < 0.01:  # 1% chance to log
                print(f"Recording waveform: {len(self.waveform_data)} samples")
        elif not self.playing:
            # If not recording or playing, reset waveform data and start time
            self.waveform_data = np.array([], dtype=np.int16)
            if hasattr(self, 'recording_start_time'):
                delattr(self, 'recording_start_time')
            if hasattr(self, 'recording_duration'):
                delattr(self, 'recording_duration')
        
    def stop_recording(self):
        """Stop recording audio"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.recording = False
        
    def save_recording(self, file_path=None, file_format='wav'):
        """Save recorded audio to file"""
        if not self.frames:
            print("No recording frames to save")
            return False
            
        # If no file_path is provided, use a default name
        if not file_path:
            # Create a default file path in user's Documents folder
            docs_dir = os.path.join(os.path.expanduser("~"), "Documents")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(docs_dir, f"recording_{timestamp}.{file_format}")
            print(f"Using default file path: {file_path}")
            
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Add file extension if not provided
        if not file_path.lower().endswith(('.' + file_format)):
            file_path += '.' + file_format
        
        # Combine all frames
        combined_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        
        try:
            if file_format == 'wav':
                # Save directly as WAV
                wf = wave.open(file_path, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pa.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(combined_data.tobytes())
                wf.close()
                print(f"Saved WAV file: {file_path}")
            else:
                # Save as WAV first, then convert to MP3
                temp_wav = file_path.replace('.' + file_format, '.wav')
                wf = wave.open(temp_wav, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pa.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(combined_data.tobytes())
                wf.close()
                
                # Convert WAV to MP3
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(file_path, format=file_format)
                print(f"Saved {file_format.upper()} file: {file_path}")
                
                # Remove temporary WAV file
                os.remove(temp_wav)
                
            return file_path
        except Exception as e:
            print(f"Error saving audio file: {e}")
            traceback.print_exc()
            return False
        
    def load_audio_file(self, file_path):
        """Load an audio file for playback and visualization
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False
            
        try:
            # Create sound object for playback
            self.current_audio = SoundLoader.load(file_path)
            
            if not self.current_audio:
                print(f"Could not load sound: {file_path}")
                return False
                
            print(f"Sound loaded: {self.current_audio.source}, Duration: {self.current_audio.length:.2f}s")
            
            # Reset playback state
            self.playback_position = 0
            self.playback_progress = 0
            self.playing = False
            self.paused = False
            
            # Get the sample rate if available
            self.sample_rate = getattr(getattr(self.current_audio, 'source', None), 'rate', 44100)
            print(f"Sample rate: {self.sample_rate}")
            
            # Load audio data for visualization
            try:
                # Extract waveform data from the audio file
                raw_waveform_data = None
                
                if file_path.lower().endswith('.wav'):
                    print("Loading WAV file for visualization...")
                    with wave.open(file_path, 'rb') as wf:
                        self.sample_rate = wf.getframerate()
                        n_frames = wf.getnframes()
                        n_channels = wf.getnchannels()
                        
                        # Store the original file's sample rate
                        self.file_sample_rate = self.sample_rate
                        
                        # Check if this is a PhysioNet heart sound recording (typically at 2000 Hz)
                        filename = os.path.basename(file_path)
                        if (filename[0].lower() in ['a', 'b', 'c', 'd', 'e'] and 
                            filename[1:].isdigit() and 
                            abs(self.sample_rate - 2000) < 100):
                            print(f"PhysioNet heart sound recording detected: {filename} at {self.sample_rate} Hz")
                        
                        print(f"WAV info - frames: {n_frames}, rate: {self.sample_rate}, channels: {n_channels}")
                        raw_waveform_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                        # If stereo, convert to mono by averaging channels
                        if n_channels == 2:
                            raw_waveform_data = raw_waveform_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                else:  # MP3 or other formats
                    print("Loading MP3/other format for visualization...")
                    # Use pydub to convert any format to raw audio data
                    audio = AudioSegment.from_file(file_path)
                    self.sample_rate = audio.frame_rate
                    print(f"AudioSegment info - channels: {audio.channels}, rate: {self.sample_rate}, samples: {len(audio.get_array_of_samples())}")
                    # Convert to mono for consistent visualization
                    audio = audio.set_channels(1)
                    raw_waveform_data = np.array(audio.get_array_of_samples())
                
                # Make a copy of the original waveform for visualization
                if raw_waveform_data is not None and len(raw_waveform_data) > 0:
                    self.waveform_data = raw_waveform_data.copy()
                    print(f"Loaded waveform data: {len(self.waveform_data)} samples, range: {np.min(self.waveform_data)} to {np.max(self.waveform_data)}")
                else:
                    print("No waveform data was loaded")
                    
            except Exception as e:
                print(f"Error loading waveform data: {e}")
                traceback.print_exc()
            
            # Reset the waveform plot to ensure it starts blank
            # This will be done by the main app class in the GUI thread
            
            # Analyze the heart sound after loading (if it's a heart sound file)
            if hasattr(self, 'heart_sound_classifier'):
                try:
                    # Store the file path for future reference
                    self.current_file_path = file_path
                    
                    # Optionally perform analysis now or wait for user to request it
                    # self.analyze_heart_sound(file_path)
                except Exception as e:
                    print(f"Error in automatic heart sound analysis: {e}")
                    # Continue even if analysis fails
            
            return True
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            traceback.print_exc()
            return False
        
    def play_audio(self):
        """Play the loaded audio file and automatically analyze the heart sound
        
        Returns:
            bool: True if playback started, False otherwise
        """
        if not self.current_audio:
            print("No audio loaded to play")
            return False
            
        try:
            # Start or resume playback
            if not self.playing:
                self.playing = True
                self.paused = False
                
                # If this is a new playback or we're starting from the beginning
                if self.current_audio.state == 'stop' or self.current_audio.get_pos() <= 0:
                    # Reset position tracking
                    self.playback_position = 0
                    self.playback_progress = 0
                    self.play_start_time = time.time()
                    self.accumulated_pause_time = 0
                    
                    # Make sure we have waveform data from the audio file
                    if (not hasattr(self, 'waveform_data') or 
                        self.waveform_data is None or 
                        len(self.waveform_data) == 0):
                        
                        try:
                            # Extract waveform data from the audio file if we don't have it
                            self.extract_waveform_from_audio()
                        except Exception as e:
                            print(f"Warning: Failed to extract waveform data: {e}")
                            traceback.print_exc()
                            # Continue playback even if we fail to extract waveform
                            # Just create an empty waveform array so other code doesn't crash
                    
                    # Start heart sound analysis in background as soon as we play
                    # We'll get the result when playback finishes
                    if hasattr(self, 'current_file_path') and self.current_file_path:
                        # Start analysis in the background without blocking playback
                        Clock.schedule_once(lambda dt: self.start_background_analysis(), 0.1)
                    
                    # Create empty waveform data if extraction failed
                    if not hasattr(self, 'waveform_data') or self.waveform_data is None or len(self.waveform_data) == 0:
                        self.waveform_data = np.zeros(1000, dtype=np.int16)
                        
                    # Start playback with error handling
                    try:
                        self.current_audio.play()
                        print("Started audio playback from beginning")
                    except Exception as e:
                        print(f"Error starting playback: {e}")
                        traceback.print_exc()
                        self.playing = False
                        return False
                else:
                    # Resuming from pause
                    resume_time = time.time()
                    # Track accumulated pause time
                    if hasattr(self, 'pause_time') and self.pause_time > 0:
                        self.accumulated_pause_time += (resume_time - self.pause_time)
                    
                    # Resume playback with error handling
                    try:
                        self.current_audio.play()
                        print(f"Resumed audio playback, paused for {resume_time - self.pause_time:.2f}s")
                    except Exception as e:
                        print(f"Error resuming playback: {e}")
                        traceback.print_exc()
                        self.playing = False
                        return False
                
                # Print audio state info
                print(f"Audio state: {self.current_audio.state}, duration: {self.current_audio.length:.2f}s")
                print(f"Starting position tracking with start_time={self.play_start_time}")
                
                return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            traceback.print_exc()
            
        return False
        
    def extract_waveform_from_audio(self):
        """Extract waveform data from the current audio file for visualization"""
        try:
            if not self.current_audio or not hasattr(self.current_audio, 'source'):
                print("No audio file available for extracting waveform data")
                return
                
            print(f"Extracting waveform data from {self.current_audio.source}")
            
            # Use scipy to read the audio file
            try:
                import scipy.io.wavfile as wavfile
                sample_rate, audio_data = wavfile.read(self.current_audio.source)
                
                # If stereo, convert to mono by averaging channels
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1).astype(np.int16)
                    
                # Store the data for waveform display
                self.waveform_data = audio_data
                self.sample_rate = sample_rate
                
                print(f"Extracted waveform: {len(audio_data)} samples at {sample_rate}Hz")
                return
            except Exception as scipy_error:
                print(f"Error with scipy: {scipy_error}")
            
            # Fallback to pydub if scipy failed
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(self.current_audio.source)
                
                # Convert to numpy array
                audio_data = np.array(audio.get_array_of_samples())
                
                # If stereo, convert to mono
                if audio.channels == 2:
                    audio_data = np.array(audio_data.reshape((-1, 2)).mean(axis=1), dtype=np.int16)
                    
                # Store the data for waveform display
                self.waveform_data = audio_data
                self.sample_rate = audio.frame_rate
                
                print(f"Extracted waveform with pydub: {len(audio_data)} samples at {audio.frame_rate}Hz")
                return
            except Exception as pydub_error:
                print(f"Error with pydub: {pydub_error}")
                
            # Last resort - generate dummy data
            print("Using dummy waveform data")
            dummy_duration = self.current_audio.length if hasattr(self.current_audio, 'length') else 10
            sample_rate = 44100
            self.sample_rate = sample_rate
            self.waveform_data = np.zeros(int(sample_rate * dummy_duration), dtype=np.int16)
            
        except Exception as e:
            print(f"Error extracting waveform data: {e}")
            traceback.print_exc()
        
    def pause_audio(self):
        """Pause the currently playing audio"""
        if self.playing and self.current_audio:
            self.current_audio.stop()
            self.playing = False
            self.paused = True
            
            # Record the pause time for timing calculations
            self.pause_time = time.time()
            print(f"Paused audio at {self.pause_time}, current position: {self.get_playback_position():.2f}s")
            return True
        return False
        
    def resume_audio(self):
        """Resume previously paused audio"""
        if self.paused and self.current_audio:
            self.playing = True
            self.paused = False
            
            # Note resume time to calculate pause duration
            resume_time = time.time()
            # Track accumulated pause time
            if self.pause_time > 0:
                self.accumulated_pause_time += (resume_time - self.pause_time)
            self.current_audio.play()
            print(f"Resumed audio playback, paused for {resume_time - self.pause_time:.2f}s")
            return True
        return False
        
    def stop_audio(self):
        """Stop the currently playing audio"""
        if self.current_audio:
            self.current_audio.stop()
            self.playing = False
            self.paused = False
            # Reset position tracking
            self.playback_position = 0
            self.playback_progress = 0
            self.play_start_time = 0
            self.pause_time = 0
            self.accumulated_pause_time = 0
            print("Stopped audio playback")
            return True
        return False
        
    def get_audio_duration(self):
        """Get duration of the current audio file"""
        if self.current_audio:
            # Get length from audio object with a minimum valid duration
            length = self.current_audio.length
            if length <= 0 and hasattr(self, 'waveform_data') and self.sample_rate > 0:
                # Fallback to calculating from waveform data
                length = len(self.waveform_data) / self.sample_rate
                print(f"Calculated duration from waveform: {length:.2f}s")
            
            # Ensure we have a reasonable minimum duration to prevent instant ending
            if length < 0.5:
                print(f"Warning: Very short audio duration detected: {length}s, setting minimum of 1s")
                length = 1.0
                
            return length
        return 0
        
    def update_playback_position(self, dt):
        """Update the current playback position based on time elapsed"""
        if self.playing and self.current_audio:
            try:
                # First try the built-in get_pos method 
                audio_pos = self.current_audio.get_pos()
                
                # If we get a valid position from get_pos, use it
                if audio_pos > 0 and audio_pos <= self.current_audio.length:
                    # Convert audio position (in seconds) to sample index
                    self.playback_position = int(audio_pos * self.sample_rate)
                    # Calculate progress for UI indication (0-100%)
                    self.playback_progress = audio_pos / self.current_audio.length
                    
                    # Occasionally log position when using get_pos
                    if random.random() < 0.02:  # Reduce log frequency to ~0.5% of updates
                        print(f"Using get_pos(): {audio_pos:.2f}s / {self.current_audio.length:.2f}s ({self.playback_progress*100:.1f}%)")
                else:
                    # Fallback to time-based position calculation if get_pos returns 0 or invalid value
                    if self.play_start_time > 0 and self.current_audio.length > 0:
                        # Calculate elapsed time since playback started
                        current_time = time.time()
                        elapsed = current_time - self.play_start_time - self.accumulated_pause_time
                        
                        # Ensure we don't exceed the audio duration
                        elapsed = min(elapsed, self.current_audio.length)
                        
                        # Calculate progress as percentage of audio duration
                        self.playback_progress = elapsed / self.current_audio.length
                        if self.playback_progress > 1.0:
                            self.playback_progress = 1.0
                            
                        # Convert to sample position for waveform display
                        self.playback_position = int(self.playback_progress * len(self.waveform_data))
                        
                        # Occasionally log position for time-based tracking
                        if random.random() < 0.02:
                            print(f"Using time-based: {elapsed:.2f}s / {self.current_audio.length:.2f}s ({self.playback_progress*100:.1f}%)")
                
                # Check if playback has ended
                if self.playback_progress >= 0.99:  # Consider it ended at 99%
                    print("Playback appears to have reached the end")
                    if self.current_audio.state != 'play':
                        self.playing = False
                        print("Auto-stopping at end of playback")
                
            except Exception as e:
                print(f"Error updating playback position: {e}")
                traceback.print_exc()
                
    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.pa.terminate()
        
    def analyze_heart_sound(self, file_path=None):
        """Analyze heart sound recording using the 1D CNN model
        
        Args:
            file_path: Path to the audio file to analyze. If None, uses the current waveform data.
            
        Returns:
            Dictionary with analysis results or None if analysis failed
        """
        try:
            # If file path is provided, use that file
            if file_path:
                self.current_file_path = file_path
                print(f"Analyzing heart sound from file: {file_path}")
                result = self.heart_sound_classifier.predict(file_path=file_path)
            # Otherwise use current waveform data if available
            elif hasattr(self, 'waveform_data') and len(self.waveform_data) > 0:
                # Use the file_sample_rate if it exists, otherwise use the current sample_rate
                fs = getattr(self, 'file_sample_rate', self.sample_rate)
                print(f"Analyzing heart sound from waveform data with sampling rate: {fs} Hz")
                
                result = self.heart_sound_classifier.predict(
                    audio_data=self.waveform_data, 
                    fs=fs
                )
            else:
                print("No audio data available for analysis")
                return None
                
            # Store the analysis result
            self.analysis_result = result
            print(f"Heart sound analysis result: {result['prediction']} (confidence: {result['confidence']:.2f})")
            return result
        except Exception as e:
            print(f"Error analyzing heart sound: {e}")
            traceback.print_exc()
            return None
            
    def generate_explanation(self, file_path=None):
        """Generate SHAP explanation for heart sound analysis
        
        Args:
            file_path: Path to the audio file to analyze. If None, uses current waveform data.
            
        Returns:
            SHAP explanation values or None if analysis failed
        """
        try:
            # If file path is provided, use that file
            if file_path:
                print(f"Generating explanation for file: {file_path}")
                explanation = self.heart_sound_classifier.get_explanation(file_path=file_path)
            # Otherwise use current waveform data if available
            elif hasattr(self, 'waveform_data') and len(self.waveform_data) > 0:
                # Use the file_sample_rate if it exists, otherwise use the current sample_rate
                fs = getattr(self, 'file_sample_rate', self.sample_rate)
                print(f"Generating explanation from waveform data with sampling rate: {fs} Hz")
                
                explanation = self.heart_sound_classifier.get_explanation(
                    audio_data=self.waveform_data,
                    fs=fs
                )
            else:
                print("No audio data available for explanation")
                return None
                
            return explanation
        except Exception as e:
            print(f"Error generating explanation: {e}")
            traceback.print_exc()
            return None
        
class WaveformPlot(FigureCanvasKivyAgg):
    """Class for plotting waveform data"""
    
    def __init__(self, controller, **kwargs):
        """Initialize the waveform plot widget"""
        # Create a Figure and set dark theme
        plt.style.use('dark_background')  # Use dark theme
        
        # Check if we're on mobile to adjust figure size
        self.is_mobile = platform in ('android', 'ios')
        
        # Adjust figure size and DPI based on platform
        if self.is_mobile:
            # Smaller figure, higher DPI for mobile
            figsize = (6, 2)
            dpi = 120
            line_width = 0.8
        else:
            # Larger figure for desktop
            figsize = (8, 3)
            dpi = 100
            line_width = 1
        
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.fig.patch.set_facecolor('#1E1E1E')  # Dark gray background
        
        # Initialize figure canvas with our figure
        super(WaveformPlot, self).__init__(figure=self.fig, **kwargs)
        
        # Store reference to containing widget
        self.controller = controller
        self.is_progressive_drawing = True  # Set to True for progressive drawing
        self.current_visible_samples = 0  # Track how many samples we are showing
        self.update_enabled = False  # Flag to control whether updates actually modify the plot
        
        # Initialize empty plot with green line
        self.line, = self.ax.plot([], [], lw=line_width, color='#00FF00')  # Bright green line
        
        # Set up a red vertical line for marking playback position
        self.playback_marker, = self.ax.plot([0, 0], [-32768, 32768], 'r-', lw=line_width, alpha=0.7)
        self.playback_marker.set_visible(False)  # Hide initially
        
        # Add timestamp text display in top-right corner
        font_size = 8 if self.is_mobile else 10
        self.timestamp_text = self.ax.text(0.98, 0.95, "00:00.00 / 00:00.00", 
                                          transform=self.ax.transAxes,
                                          color='white', fontsize=font_size,
                                          horizontalalignment='right')
        
        # Set up plot styling
        label_size = 8 if self.is_mobile else 10
        tick_size = 7 if self.is_mobile else 9
        
        self.ax.set_xlabel('Time (seconds)', color='#AAAAAA', fontsize=label_size)
        self.ax.set_ylabel('Amplitude', color='#AAAAAA', fontsize=label_size)
        self.ax.tick_params(axis='both', colors='#AAAAAA', labelsize=tick_size)
        self.ax.grid(True, alpha=0.3, color='#444444')
        
        # Start with reasonable y-axis limits for audio data
        self.ax.set_ylim(-32768, 32768)
        
        # Set x-axis to display time in seconds format
        self.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}s"))
        
        # Initial x-axis limits - start with a 10-second window
        self.ax.set_xlim(0, 10)
        
        # Adjust layout for better appearance
        self.fig.tight_layout(pad=1.5 if self.is_mobile else 2.0)
        
        # Start with blank data
        self.reset_for_playback()
        
        # Schedule regular updates to the plot, but at a lower rate initially
        # We don't want to draw anything until the user starts playback
        Clock.schedule_interval(self.update_plot, 1/10)  # 10 FPS updates
        
    def reset_for_playback(self):
        """Reset the plot for a new playback animation"""
        # Clear existing plot data
        self.line.set_data(np.array([]), np.array([]))
        
        # Hide playback marker
        self.playback_marker.set_visible(False)
        
        # Reset timestamp
        self.timestamp_text.set_text("00:00.00 / 00:00.00")
        
        # Reset visible samples counter
        self.current_visible_samples = 0
        
        # Force redraw
        self.draw_idle()
        
        # Start with updates disabled - will be enabled when playback starts
        self.update_enabled = False
        
    def draw_idle(self):
        """Force a redraw of the canvas"""
        self.draw()  # Since we're a FigureCanvasKivyAgg, we call draw directly
        
    def update_plot(self, dt):
        """Update the waveform plot display
        
        Args:
            dt (float): Delta time since last update
        """
        # If updates are disabled, don't modify the plot
        if not self.update_enabled:
            return
            
        if not hasattr(self.controller, 'waveform_data') or self.controller.waveform_data is None:
            # Only log occasionally when debugging
            if random.random() < 0.005:  # Reduce log frequency to ~0.5% of updates
                print("No waveform data available")
            return
            
        # Get data from controller
        full_data = self.controller.waveform_data
        
        if len(full_data) == 0:
            # Only log occasionally when debugging
            if random.random() < 0.005:  # Reduce log frequency
                print("Waveform data is empty")
            
            # Keep the plot blank if there's no data
            self.line.set_data(np.array([]), np.array([]))
            self.playback_marker.set_visible(False)
            self.draw_idle()
            return
            
        # Check if we're in recording mode
        is_recording = (hasattr(self.controller, 'recording') and self.controller.recording)
        
        # Check if we're in progressive drawing mode during playback
        is_playback = (hasattr(self.controller, 'playing') and self.controller.playing)
        
        # Get playback position and audio info
        playback_position = 0
        playback_progress = 0
        audio_duration = 0
        sample_rate = getattr(self.controller, 'sample_rate', 44100)
        
        # Debug information occasionally
        if random.random() < 0.005:
            print(f"Plot update: recording={is_recording}, playback={is_playback}, data_len={len(full_data)}")
            
        if is_playback:
            playback_position = getattr(self.controller, 'playback_position', 0)
            playback_progress = getattr(self.controller, 'playback_progress', 0)
            
            # Get audio duration
            if hasattr(self.controller, 'current_audio') and self.controller.current_audio:
                audio_duration = self.controller.current_audio.length
                
                # Update timestamp text
                timestamp = f"{format_time(playback_progress * audio_duration)} / {format_time(audio_duration)}"
                self.timestamp_text.set_text(timestamp)
                
                # Print debug info occasionally
                if random.random() < 0.01:
                    print(f"Playback: {playback_progress * audio_duration:.2f}s / {audio_duration:.2f}s ({playback_progress*100:.1f}%)")
                
                # Progressive drawing during playback
                if self.is_progressive_drawing and len(full_data) > 0:
                    # Calculate how many samples to show based on playback progress
                    target_visible = int(playback_progress * len(full_data))
                    target_visible = max(10, min(len(full_data), target_visible))
                    
                    # Store the current number of visible samples
                    self.current_visible_samples = target_visible
                    
                    # Get visible portion of data
                    visible_data = full_data[:target_visible]
                    
                    # Downsample data if needed for display efficiency
                    max_points = 2000  # Increase for higher resolution
                    if len(visible_data) > max_points:
                        indices = np.linspace(0, len(visible_data) - 1, max_points).astype(int)
                        plot_data = visible_data[indices]
                        
                        # Scale time values based on audio duration
                        plot_time = np.linspace(0, playback_progress * audio_duration, len(plot_data))
                    else:
                        plot_data = visible_data
                        # Scale time to actual seconds elapsed
                        plot_time = np.linspace(0, playback_progress * audio_duration, len(plot_data))
                    
                    # Update plot data (x=time in seconds, y=amplitude)
                    self.line.set_data(plot_time, plot_data)
                    
                    # Update x-axis limits to show a bit ahead of current position
                    if playback_progress * audio_duration > 0:
                        # Show a window that adapts to the current position
                        window_size = min(10, max(2, audio_duration / 5))  # Adjust as needed
                        
                        # Set left edge at 0 or slightly to the left of current time
                        left_edge = max(0, playback_progress * audio_duration - window_size * 0.3)
                        
                        # Set right edge to show a window ahead
                        right_edge = left_edge + window_size
                        
                        # Never show beyond the audio duration
                        if right_edge > audio_duration:
                            right_edge = audio_duration
                            left_edge = max(0, right_edge - window_size)
                            
                        self.ax.set_xlim(left_edge, right_edge)
                    
                    # Draw the playback position marker at current time
                    self.playback_marker.set_xdata([playback_progress * audio_duration, playback_progress * audio_duration])
                    self.playback_marker.set_ydata([-32768, 32768])
                    self.playback_marker.set_visible(True)
                    
                    # Ensure x-axis shows time in seconds
                    if random.random() < 0.05:  # Occasionally update ticks
                        self.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}s"))
                    
                    # Force redraw
                    self.draw_idle()
                else:
                    # Non-progressive playback mode (full waveform)
                    # Just update the playback marker position
                    if len(plot_time) > 0:
                        self.playback_marker.set_xdata([playback_progress * audio_duration, playback_progress * audio_duration])
                        self.playback_marker.set_visible(True)
                        self.draw_idle()
            
        elif is_recording:
            # For recording, get the real recording duration from controller
            recorded_duration = 0
            if hasattr(self.controller, 'recording_duration'):
                recorded_duration = self.controller.recording_duration
            else:
                # Fallback to calculating from samples and sample rate
                recorded_duration = len(full_data) / sample_rate
                
            # Set timestamp for recording display
            timestamp = f"{format_time(recorded_duration)} / Recording"
            self.timestamp_text.set_text(timestamp)
            
            # Progressive drawing during recording - we want to show all data from beginning
            if len(full_data) > 0:
                # Debug log occasionally
                if random.random() < 0.01:
                    print(f"Recording graph: duration={recorded_duration:.2f}s, samples={len(full_data)}")
                
                # Always use the full recorded data
                visible_data = full_data
                
                # Downsample data if needed for efficient display
                max_points = 2000  # Max points to show for smooth rendering
                if len(visible_data) > max_points:
                    # Sample points evenly across the full dataset
                    indices = np.linspace(0, len(visible_data) - 1, max_points).astype(int)
                    plot_data = visible_data[indices]
                    # Use actual time values (left to right)
                    plot_time = np.linspace(0, recorded_duration, len(plot_data))
                else:
                    # Use all data points if we have fewer than max_points
                    plot_data = visible_data
                    # Use actual time values (left to right)
                    plot_time = np.linspace(0, recorded_duration, len(plot_data))
                
                # Update plot data - this maps x=time and y=amplitude
                self.line.set_data(plot_time, plot_data)
                
                # Calculate window for scrolling display during recording
                window_size = 5  # Show 5 seconds of recording
                
                # Scrolling window: keep latest data visible and scroll from left to right
                if recorded_duration > window_size:
                    # We're past the initial window - scroll to keep latest data visible
                    left_edge = recorded_duration - window_size  # Keep the window size consistent
                    right_edge = recorded_duration + 0.5  # Add small margin on right
                else:
                    # Initial recording - start at 0 and expand
                    left_edge = 0
                    right_edge = max(window_size, recorded_duration + 1)
                
                # Set the axis limits for the scrolling window
                self.ax.set_xlim(left_edge, right_edge)
                
                # Show current position marker at the recording edge
                self.playback_marker.set_xdata([recorded_duration, recorded_duration])
                self.playback_marker.set_ydata([-32768, 32768])
                self.playback_marker.set_visible(True)
                
                # Force redraw
                self.draw_idle()
        else:
            # Non-progressive mode (show full waveform with time-based x-axis)
            if len(full_data) > 0:
                # Calculate time values based on sample rate
                if hasattr(self.controller, 'current_audio') and self.controller.current_audio:
                    audio_duration = self.controller.current_audio.length
                    
                    # Downsample if needed
                    max_points = 2000
                    if len(full_data) > max_points:
                        indices = np.linspace(0, len(full_data) - 1, max_points).astype(int)
                        plot_data = full_data[indices]
                        plot_time = np.linspace(0, audio_duration, len(plot_data))
                    else:
                        plot_data = full_data
                        plot_time = np.linspace(0, audio_duration, len(plot_data))
                    
                    # Update plot data
                    self.line.set_data(plot_time, plot_data)
                    
                    # Set x-axis to show full duration
                    self.ax.set_xlim(0, audio_duration)
                    
                    # Update timestamp
                    self.timestamp_text.set_text(f"00:00.00 / {format_time(audio_duration)}")
                    
                    # Hide playback marker when not playing
                    self.playback_marker.set_visible(False)
                    
                    # Force redraw
                    self.draw_idle()
            
        # Auto-scale y-axis for better visualization
        y_data = self.line.get_ydata()
        if len(y_data) > 0:
            # Convert to numpy array if it's a list
            if isinstance(y_data, list):
                y_data = np.array(y_data)
                
            y_max = np.max(np.abs(y_data))
            if y_max > 100:  # Only adjust if there's meaningful data
                # Set y limits with some padding
                new_limit = min(32768, max(1000, y_max * 1.2))
                self.ax.set_ylim(-new_limit, new_limit)
        
        # Force redraw of the canvas
        self.draw_idle()

    def fit_to_data(self):
        """Automatically fit the plot to the data range"""
        if not hasattr(self.controller, 'waveform_data') or self.controller.waveform_data is None:
            print("No waveform data to fit to")
            return
            
        # Get data from controller
        full_data = self.controller.waveform_data
        
        if len(full_data) == 0:
            print("Waveform data is empty, cannot fit")
            return
            
        # Get sample rate from controller
        sample_rate = getattr(self.controller, 'sample_rate', 44100)
        
        # Calculate duration in seconds
        duration = len(full_data) / sample_rate
        
        # Ensure duration is at least 1 second for visibility
        duration = max(1.0, duration)
        
        # Set x-axis to show the full duration
        self.ax.set_xlim(0, duration)
        
        # Set y-axis to show a bit above and below the min/max values
        if len(full_data) > 0:
            data_min = np.min(full_data)
            data_max = np.max(full_data)
            y_margin = (data_max - data_min) * 0.1  # 10% margin
            
            # Ensure we have some range even for constant data
            if data_max == data_min:
                if data_max == 0:
                    # For silent recording
                    self.ax.set_ylim(-1000, 1000)
                else:
                    # For constant non-zero values
                    self.ax.set_ylim(data_min * 0.9, data_min * 1.1)
            else:
                # Normal case with varying data
                self.ax.set_ylim(data_min - y_margin, data_max + y_margin)
        else:
            # Default y-axis limits if we somehow have no data
            self.ax.set_ylim(-32768, 32768)
        
        # Update timestamp text to show duration
        if duration > 0:
            self.timestamp_text.set_text(f"Duration: {format_time(duration)}")
            
        # Force redraw
        self.draw_idle()
        print(f"Plot fitted to data range: {duration:.2f} seconds")

# Helper function for timestamp formatting
def format_time(seconds):
    """Format seconds as MM:SS.ms"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

class SaveDialog(BoxLayout):
    """Dialog content for saving audio files"""
    def __init__(self, **kwargs):
        super(SaveDialog, self).__init__(**kwargs)
        self.size_hint_y = None
        self.height = "170dp"  # Increased height for the directory selector
        self.selected_directory = os.path.join(os.path.expanduser("~"), "Documents")

class AudioRecorderApp(MDApp):
    """Main application class for audio recorder"""
    is_recording = BooleanProperty(False)
    is_playing = BooleanProperty(False)
    enable_stop = BooleanProperty(False)
    timer_text = StringProperty("00:00.00")
    is_mobile = BooleanProperty(False)  # Flag to detect if running on mobile
    analysis_result = StringProperty("Not analyzed")
    analysis_confidence = NumericProperty(0)
    analysis_color = StringProperty("#888888")  # Default gray color

    def __init__(self, **kwargs):
        """Initialize the application class"""
        super(AudioRecorderApp, self).__init__(**kwargs)
        self.controller = None
        self.waveform_plot = None
        self.timer_event = None
        self.timer_seconds = 0

        # Detect platform
        self.is_mobile = platform in ('android', 'ios')
        print(f"Running on platform: {platform}, is_mobile: {self.is_mobile}")

        # Initialize UI elements
        self.controller = AudioController()
        self.waveform_plot = WaveformPlot(self.controller)

    def build(self):
        """Build the application UI"""
        # Set app theme
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.primary_hue = "700"
        self.theme_cls.accent_palette = "Teal"
        
        # Set app title and icon
        self.title = "Rakib FYP XAI Audio Phonogram Monitor"
        self.icon = "Lovepik_com-402182829-3d-medical-series-icon-ecg.png"

        # Load Kivy file and get root widget
        self.root = Builder.load_file("audio_recorder.kv")

        return self.root

    def on_start(self):
        """Events to run when the application starts"""
        # Create file manager
        self.file_manager = MDFileManager(
            exit_manager=self.exit_file_manager,
            select_path=self.select_file_path,
        )

        # We'll add the waveform plot to the container in on_start
        # because self.root is not available yet
        waveform_container = self.root.ids.waveform_container
        if hasattr(self, 'waveform_plot') and self.waveform_plot:
            waveform_container.add_widget(self.waveform_plot)

            # Unschedule any existing update
            Clock.unschedule(self.waveform_plot.update_plot)
            # Schedule at 30fps for smoother display
            Clock.schedule_interval(self.waveform_plot.update_plot, 1/30)

        # Initialize audio references consistently
        if hasattr(self, 'controller'):
            self.audio_controller = self.controller  # For backwards compatibility
            
        # Make sure buttons show correct labels on start
        if hasattr(self.root.ids, 'play_button'):
            self.root.ids.play_button.text = "Play"
        if hasattr(self.root.ids, 'pause_button'):
            self.root.ids.pause_button.text = "Pause"

    def update_timer(self, dt):
        """Update the timer display"""
        if self.is_playing and hasattr(self, 'controller') and self.controller:
            # If audio is playing, try to get actual position from audio controller
            if hasattr(self.controller, 'current_audio') and self.controller.current_audio:
                try:
                    # Get position directly from audio object for better accuracy
                    audio_pos = self.controller.current_audio.get_pos()
                    audio_duration = self.controller.get_audio_duration()
                    
                    # If we get a valid position from get_pos, use it
                    if audio_pos > 0 and audio_pos <= audio_duration:
                        # Use actual audio position for timer
                        self.timer_seconds = audio_pos
                    else:
                        # Fallback to incremental timer if get_pos returns 0 or negative
                        self.timer_seconds += dt
                except Exception as e:
                    # If any error getting position, use incremental timer
                    self.timer_seconds += dt
            else:
                # No audio object, use incremental timer
                self.timer_seconds += dt
        elif self.is_recording:
            # For recording, increment the timer
            self.timer_seconds += dt
        
        # Format and display the time
        minutes = int(self.timer_seconds) // 60
        seconds = int(self.timer_seconds) % 60
        centiseconds = int((self.timer_seconds - int(self.timer_seconds)) * 100)
        self.timer_text = f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
        
    def update_timer_display(self):
        """Reset and display timer at zero"""
        self.timer_seconds = 0
        minutes = 0
        seconds = 0
        centiseconds = 0
        self.timer_text = f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def start_recording(self):
        """Start recording audio"""
        if not self.is_recording and not self.is_playing:
            # Reset recording timer
            self.timer_seconds = 0
            self.timer_text = "00:00.00"
            
            # Initialize waveform plot if not done already
            if not hasattr(self, 'waveform_plot') or not self.waveform_plot:
                self.initialize_waveform_plot()
                
            # Configure waveform plot for recording
            if hasattr(self, 'waveform_plot') and self.waveform_plot:
                self.waveform_plot.reset_for_playback()  # Reset to clear any previous data
                self.waveform_plot.update_enabled = True  # Enable updates
                self.waveform_plot.is_progressive_drawing = True  # Show progressive recording
                
                # Set initial x-axis range for recording (0 to 5 seconds)
                self.waveform_plot.ax.set_xlim(0, 5)
                self.waveform_plot.draw_idle()
                
                # Schedule more frequent updates during recording for smooth display
                Clock.unschedule(self.waveform_plot.update_plot)
                Clock.schedule_interval(self.waveform_plot.update_plot, 1/30)  # 30 fps updates
                
            try:
                # Start the audio recording with the controller
                if self.controller.start_recording():
                    self.is_recording = True
                    self.is_playing = False
                    
                    # Update the UI state
                    self.update_file_label(mode='recording')
                    self.root.ids.record_button.text = "Stop"
                    
                    # Start the timer for recording duration
                    if hasattr(self, 'timer_event') and self.timer_event:
                        Clock.unschedule(self.timer_event)
                    self.timer_event = Clock.schedule_interval(self.update_timer, 0.01)  # 100 fps timer
                    
                    # Schedule regular checking of the recording waveform
                    self.waveform_update_event = Clock.schedule_interval(
                        self.update_recording_waveform, 0.05)  # 20 fps waveform updates
                    
                    return True
                else:
                    self.show_message("Failed to start recording")
            except Exception as e:
                print(f"Error starting recording: {e}")
                traceback.print_exc()
                self.show_message(f"Error starting recording: {str(e)}")
                
        return False
        
    def stop_recording(self):
        """Stop recording audio"""
        if self.is_recording:
            try:
                # Stop controller recording
                if hasattr(self, 'controller') and self.controller:
                    self.controller.stop_recording()
                
                # Update UI state
                self.is_recording = False
                self.root.ids.record_button.text = "Record"
                
                # Unschedule the timer if it exists
                if hasattr(self, 'timer_event') and self.timer_event:
                    Clock.unschedule(self.timer_event)
                    self.timer_event = None
                
                # Unschedule waveform updates 
                if hasattr(self, 'waveform_update_event') and self.waveform_update_event:
                    Clock.unschedule(self.waveform_update_event)
                    self.waveform_update_event = None
                
                # Reset waveform plot update frequency to normal
                if hasattr(self, 'waveform_plot') and self.waveform_plot:
                    Clock.unschedule(self.waveform_plot.update_plot)
                    Clock.schedule_interval(self.waveform_plot.update_plot, 1/10)  # Back to 10 fps updates
                    # Keep update_enabled active to show the full recording
                
                # First save a temporary recording for immediate visualization
                if hasattr(self, 'controller') and self.controller and hasattr(self.controller, 'frames') and self.controller.frames:
                    # Create a timestamp-based filename for the temp file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_filename = os.path.join(os.path.expanduser("~"), "Documents", f"temp_recording_{timestamp}.wav")
                    
                    # Save the recording to a temporary file
                    temp_file = self.controller.save_recording(temp_filename)
                    
                    if temp_file:
                        print(f"Saved temporary recording to: {temp_file}")
                        
                        # Update the UI to show the file is ready
                        self.update_file_label(temp_file, mode='saved')
                        
                        # Load the saved recording for visualization
                        if hasattr(self.controller, 'load_audio_file') and callable(self.controller.load_audio_file):
                            if self.controller.load_audio_file(temp_file):
                                # Enable visualization of the whole recording
                                if hasattr(self, 'waveform_plot') and self.waveform_plot:
                                    self.waveform_plot.is_progressive_drawing = False  # Show full waveform
                                    self.waveform_plot.update_enabled = True
                                    self.waveform_plot.fit_to_data()  # Auto-fit to the data range
                
                # Show the save dialog to allow the user to save the recording with a custom name
                Clock.schedule_once(lambda dt: self.show_save_dialog(), 0.5)
                
                return True
            except Exception as e:
                print(f"Error stopping recording: {e}")
                traceback.print_exc()
                self.show_message(f"Error: {str(e)}")
        
        return False
        
    def toggle_recording(self):
        """Toggle recording state"""
        if self.is_recording:
            # Stop recording
            if hasattr(self, 'controller'):
                self.controller.stop_recording()

            self.is_recording = False
            self.root.ids.record_button.text = "Record"
            self.root.ids.record_button.md_bg_color = self.theme_cls.primary_color
            self.root.ids.record_button.disabled = False
            self.enable_stop = False

            if hasattr(self, 'timer_event'):
                Clock.unschedule(self.timer_event)
                self.timer_event = None

            # Show save dialog
            self.show_save_dialog()
        else:
            # Start recording
            self.timer_seconds = 0
            if hasattr(self, 'controller'):
                # Clear any existing data
                self.controller.clear()
                # Start recording
                self.controller.start_recording()

            self.is_recording = True
            self.root.ids.record_button.text = "Recording..."
            self.root.ids.record_button.md_bg_color = [1, 0, 0, 1]  # Red color for recording
            self.root.ids.record_button.disabled = True  # Disable record button during recording

            # Start timer
            self.timer_event = Clock.schedule_interval(self.update_timer, 0.1)

            # Schedule enabling the stop button after 10 seconds
            Clock.schedule_once(self.enable_stop_button, 10)

            # Force waveform display to update more frequently during recording
            if self.waveform_plot is not None:
                # Unschedule any existing update
                Clock.unschedule(self.waveform_plot.update_plot)
                # Enable updates and progressive drawing for real-time visualization
                self.waveform_plot.update_enabled = True  # IMPORTANT: Enable updates as soon as recording starts
                self.waveform_plot.is_progressive_drawing = True  # Enable progressive drawing
                # Schedule at 60fps for smoother display during recording (higher frame rate)
                Clock.schedule_interval(self.waveform_plot.update_plot, 1/60)
                print("Started waveform plotting at 60fps for recording")

    def enable_stop_button(self, dt):
        """Enable the stop button after a delay"""
        self.enable_stop = True

    def show_save_dialog(self):
        """Show dialog to save the recorded audio file"""
        if not hasattr(self, 'save_dialog') or not self.save_dialog:
            # Create dialog content
            content = SaveDialog()

            # Initialize the directory label
            content.ids.directory_label.text = self._shorten_path(content.selected_directory)

            # Create save dialog
            self.save_dialog = MDDialog(
                title="Save Audio File",
                type="custom",
                content_cls=content,
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=lambda x: self.dismiss_save_dialog()
                    ),
                    MDFlatButton(
                        text="SAVE",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=lambda x: self.save_audio_from_dialog(content)
                    ),
                ],
            )
        self.save_dialog.open()

    def dismiss_save_dialog(self):
        """Dismiss the save dialog"""
        if hasattr(self, 'save_dialog') and self.save_dialog:
            self.save_dialog.dismiss()

    def save_audio_from_dialog(self, content):
        """Save audio from dialog input"""
        try:
            # Get input values from dialog
            filename = content.ids.filename_input.text
            file_format = content.ids.format_spinner.text.lower()  # Convert to lowercase
            directory = content.selected_directory
            
            print(f"Saving audio as: {filename}, format: {file_format}, directory: {directory}")
            
            # Make sure we have a filename
            if not filename:
                self.show_message("Please enter a filename")
                return
                
            # Call the save_audio method
            success = self.save_audio(filename, file_format, directory)
            
            if success:
                self.dismiss_save_dialog()
                self.show_message(f"Audio saved as {os.path.basename(success)}")
                
                # After saving, load the saved file to display it
                if hasattr(self, 'controller') and hasattr(self.controller, 'load_audio_file'):
                    if self.controller.load_audio_file(success):
                        print(f"Loaded saved file: {success}")
                        # Update the UI
                        self.update_file_label(success, mode='saved')
                        # Update the waveform display
                        if hasattr(self, 'waveform_plot') and self.waveform_plot:
                            self.waveform_plot.fit_to_data()
            else:
                self.show_message("Failed to save audio")
        except Exception as e:
            print(f"Error saving audio from dialog: {e}")
            traceback.print_exc()
            self.show_message(f"Error: {str(e)}")

    def save_audio(self, filename, file_format='wav', directory=None):
        """Save the recorded audio

        Args:
            filename (str): Filename to save as
            file_format (str): Audio format to save as
            directory (str): Directory to save in

        Returns:
            bool: True if successful, False otherwise
        """
        if not filename:
            return False

        # Create directories if they don't exist
        if directory is None:
            docs_dir = os.path.join(os.path.expanduser("~"), "Documents")
            app_dir = os.path.join(docs_dir, "Rakib_FYP_XAI_Recordings")
        else:
            app_dir = directory

        if not os.path.exists(app_dir):
            os.makedirs(app_dir)

        # Create full path
        file_path = os.path.join(app_dir, filename)

        # Add extension if not provided
        if not file_path.lower().endswith(f'.{file_format.lower()}'):
            file_path += f'.{file_format.lower()}'

        # Check if file exists and add timestamp if it does
        if os.path.exists(file_path):
            base_name, ext = os.path.splitext(file_path)
            timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
            file_path = f"{base_name}{timestamp}{ext}"

        if hasattr(self, 'controller'):
            success = self.controller.save_recording(file_path, file_format)
            if success:
                print(f"Audio saved to: {file_path}")
                self.update_file_label(file_path, mode='saved')
                return file_path
            else:
                print("Failed to save audio")
                self.update_file_label(None)
                return False
        return False

    def open_file_manager(self):
        """Open file manager to select an audio file"""
        # Don't open file manager if recording or playing audio
        if self.is_recording or self.is_playing:
            return

        # Use different approaches for mobile vs desktop
        if self.is_mobile:
            # On mobile, we need to use plyer or a simpler approach
            if platform == 'android':
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])

            try:
                from plyer import filechooser
                filechooser.open_file(on_selection=self.handle_mobile_file_selection, 
                                     filters=[("Audio Files", "*.mp3", "*.wav")])
            except Exception as e:
                print(f"Mobile file chooser error: {e}")
                # Fallback to a basic approach for testing
                self.show_message("File selection not available on this device")
        else:
            # Desktop file manager
            documents_path = os.path.join(os.path.expanduser("~"), "Documents")
            self.file_manager.show(documents_path)

    def handle_mobile_file_selection(self, selection):
        """Handle file selection from mobile file chooser"""
        if not selection:
            print("No file selected")
            return

        # Process the selected file
        self.select_file_path(selection[0])

    def select_file_path(self, path):
        """Process the path selected from file manager"""
        # Close the file manager
        self.exit_file_manager()

        # Verify path exists
        if not path or not os.path.exists(path):
            print(f"Invalid path or file does not exist: {path}")
            return False

        # Check if it's an audio file
        if not path.lower().endswith(('.mp3', '.wav')):
            print(f"Not an audio file: {path}")
            self.show_message("Please select an MP3 or WAV file")
            return False

        file_path = os.path.abspath(path)
        print(f"Selected file: {file_path}")

        # Remember the current file path
        self.current_file = file_path

        # Load audio into the controller
        if hasattr(self.controller, 'load_audio_file') and callable(self.controller.load_audio_file):
            if self.controller.load_audio_file(file_path):
                # Create or reset waveform plot
                if not hasattr(self, 'waveform_plot') or not self.waveform_plot:
                    self.initialize_waveform_plot()
                else:
                    # Reset the waveform plot to start with a blank display
                    self.waveform_plot.reset_for_playback()
                    # Explicitly turn off updates to keep it blank until playback
                    self.waveform_plot.update_enabled = False
                    # Force a redraw to show empty canvas
                    self.waveform_plot.draw_idle()

                # Update the file label with the current file name
                self.update_file_label(file_path)

                # Show success message
                self.show_message(f"Loaded: {os.path.basename(file_path)}")

                print(f"Audio file loaded: {file_path}")
                return True
        else:
            print(f"Failed to load audio file: {file_path}")
            self.show_message("Failed to load audio file")

        return False

    def exit_file_manager(self, *args):
        """Close the file manager"""
        self.file_manager.close()

    def play_audio(self):
        """Play the loaded audio file and automatically analyze heart sound"""
        if self.is_recording:
            return False

        if not self.is_playing and self.current_file:
            try:
                # Set "Analyzing..." message to prepare user for results
                self.analysis_result = "Analyzing..."
                self.analysis_color = "#0000AA"  # Blue for processing
                
                # Make sure waveform data exists before playing
                if not hasattr(self.controller, 'waveform_data') or len(self.controller.waveform_data) == 0:
                    # Try to extract waveform data if it doesn't exist
                    if hasattr(self.controller, 'extract_waveform_from_audio'):
                        self.controller.extract_waveform_from_audio()

                # Play the audio
                if self.controller.play_audio():
                    self.is_playing = True
                    self.is_recording = False

                    # Update the label to show the currently playing file
                    self.update_file_label(self.current_file, mode='playing')

                    # Update UI button state - clearly show we're playing
                    self.root.ids.play_button.text = "Playing..."
                    self.root.ids.play_button.disabled = True
                    self.root.ids.pause_button.disabled = False
                    self.root.ids.pause_button.text = "Pause"
                    
                    # Reset and start the timer for playback tracking
                    self.timer_seconds = 0
                    if hasattr(self, 'timer_event') and self.timer_event:
                        Clock.unschedule(self.timer_event)
                    self.timer_event = Clock.schedule_interval(self.update_timer, 0.01)  # 100 fps for smooth timer

                    # Configure waveform plot for playback visualization
                    if hasattr(self, 'waveform_plot') and self.waveform_plot:
                        print("Setting up waveform display for playback...")

                        # Reset for new playback and enable updates
                        self.waveform_plot.reset_for_playback()
                        self.waveform_plot.update_enabled = True
                        self.waveform_plot.is_progressive_drawing = True

                        # Generate an immediate update to ensure the waveform is visible
                        self.waveform_plot.update_plot(0)

                        # Schedule regular updates for the waveform display
                        Clock.unschedule(self.waveform_plot.update_plot)
                        Clock.schedule_interval(self.waveform_plot.update_plot, 1/30)

                        # Schedule playback position updates for smooth marker movement
                        if hasattr(self, 'playback_position_event'):
                            Clock.unschedule(self.playback_position_event)
                        self.playback_position_event = Clock.schedule_interval(self.update_playback_position, 1/30)
                    else:
                        print("WARNING: Waveform plot not available")
                        # Try to initialize waveform plot if it doesn't exist
                        self.initialize_waveform_plot()
                        if hasattr(self, 'waveform_plot') and self.waveform_plot:
                            self.waveform_plot.reset_for_playback()
                            self.waveform_plot.update_enabled = True
                            self.waveform_plot.is_progressive_drawing = True
                            Clock.schedule_interval(self.waveform_plot.update_plot, 1/30)

                    # Also check for audio ending regularly
                    self.check_audio_ended_event = Clock.schedule_interval(self.check_audio_ended, 0.5)
                    
                    # Start heart sound analysis in background
                    start_background_analysis(self)

                    print("Started audio playback with waveform visualization and background analysis")
                    return True
            except Exception as e:
                print(f"Error playing audio: {e}")
                traceback.print_exc()
                self.show_message(f"Error playing audio: {str(e)}")
                return False

        return False
        
    def pause_audio(self):
        """Pause or resume the currently playing audio"""
        try:
            if not self.is_playing:
                return False

            if hasattr(self.controller, 'paused') and self.controller.paused:
                # If already paused, resume playback
                return self.resume_audio()
            else:
                # If playing, pause it
                if self.controller.pause_audio():
                    # We're still in "playing" state overall, just paused
                    self.is_playing = True  # Keep this true since we're in a playback session

                    # Stop timer
                    if hasattr(self, 'timer_event') and self.timer_event:
                        Clock.unschedule(self.timer_event)
                        self.timer_event = None

                    # Unschedule the playback position update
                    if hasattr(self, 'playback_position_event') and self.playback_position_event:
                        Clock.unschedule(self.playback_position_event)
                        self.playback_position_event = None

                    # Unschedule the waveform update event
                    if hasattr(self, 'waveform_update_event') and self.waveform_update_event:
                        Clock.unschedule(self.waveform_update_event)
                        self.waveform_update_event = None

                    # Update button text to "Resume"
                    if hasattr(self.root.ids, 'pause_button'):
                        self.root.ids.pause_button.text = "Resume"

                    return True
        except Exception as e:
            print(f"Error in pause_audio: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash, just return false

        return False
        
    def resume_audio(self):
        """Resume previously paused audio"""
        try:
            if self.controller.resume_audio():
                self.is_playing = True

                # Start timer
                self.timer_event = Clock.schedule_interval(self.update_timer, 0.01)

                # Schedule waveform updates at high frequency for visualization during playback
                if hasattr(self, 'waveform_update_event') and self.waveform_update_event:
                    Clock.unschedule(self.waveform_update_event)
                self.waveform_update_event = Clock.schedule_interval(self.update_waveform, 1/30)

                # Update playback position in real-time (higher frequency for smoother animation)
                if hasattr(self, 'playback_position_event') and self.playback_position_event:
                    Clock.unschedule(self.playback_position_event)
                self.playback_position_event = Clock.schedule_interval(self.update_playback_position, 1/60)

                # Disable play button, ensure pause button is enabled and shows "Pause"
                if hasattr(self.root.ids, 'play_button'):
                    self.root.ids.play_button.disabled = True
                if hasattr(self.root.ids, 'pause_button'):
                    self.root.ids.pause_button.disabled = False
                    self.root.ids.pause_button.text = "Pause"

                print("Audio playback and visualization resumed")
                return True
        except Exception as e:
            print(f"Error in resume_audio: {e}")
            import traceback
            traceback.print_exc()

        return False

    def check_audio_ended(self, dt):
        """Check if audio playback has ended"""
        try:
            # Skip check if we're in recording mode
            if self.is_recording:
                return True
                
            # Skip check if we have no controller or playing flag is already false
            if not hasattr(self, 'controller') or not self.is_playing:
                return False
                
            # Get audio state directly from the sound object
            audio_state = "unknown"
            if hasattr(self.controller, 'current_audio') and self.controller.current_audio:
                audio_state = self.controller.current_audio.state
                
            # Get current position and duration
            current_pos = 0
            if hasattr(self.controller, 'current_audio') and self.controller.current_audio:
                current_pos = self.controller.current_audio.get_pos()
                
            duration = self.controller.get_audio_duration()
            
            # Debug info occasionally
            if random.random() < 0.1:  # Only print ~10% of the time to avoid log spam
                print(f"Audio check: state={audio_state}, position={current_pos:.2f}s, duration={duration:.2f}s, controller_playing={self.controller.playing}")
            
            # Multiple conditions for end detection:
            # 1. Sound state is 'stop' (definitely stopped)
            # 2. Controller playing flag is False AND audio state is not 'play'
            # 3. Position is very close to the end AND we've been playing for at least 90% of the duration
            position_near_end = (duration > 0 and current_pos > 0 and current_pos >= 0.95 * duration)
            timer_near_end = (duration > 0 and self.timer_seconds >= 0.95 * duration)
            
            if (audio_state == 'stop' or 
                (not self.controller.playing and audio_state != 'play') or
                (position_near_end and timer_near_end)):
                
                # Definitely ended - log the reason
                if audio_state == 'stop':
                    print("Audio ended: state is 'stop'")
                elif not self.controller.playing and audio_state != 'play':
                    print(f"Audio ended: controller.playing=False, audio_state={audio_state}")
                elif position_near_end and timer_near_end:
                    print(f"Audio ended: position={current_pos:.2f}s near duration={duration:.2f}s and timer={self.timer_seconds:.2f}s")
                
                # Set playback as ended
                self.is_playing = False

                # Unschedule this callback
                Clock.unschedule(self.check_audio_ended)

                # Stop and reset timer
                if self.timer_event:
                    Clock.unschedule(self.timer_event)
                    self.timer_event = None
                
                # Reset timer display to 00:00.00 or to total duration
                if duration > 0:
                    # Show the total duration as the final time
                    minutes = int(duration) // 60
                    seconds = int(duration) % 60
                    centiseconds = int((duration - int(duration)) * 100)
                    self.timer_text = f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
                    # After a delay, reset to 00:00.00
                    Clock.schedule_once(lambda dt: setattr(self, 'timer_text', "00:00.00"), 1)
                else:
                    # Immediately reset to 00:00.00
                    self.timer_text = "00:00.00"

                # Unschedule the playback position update
                if hasattr(self, 'playback_position_event') and self.playback_position_event:
                    Clock.unschedule(self.playback_position_event)
                    self.playback_position_event = None

                # Reset UI
                if hasattr(self.root.ids, 'play_button'):
                    self.root.ids.play_button.text = "Play"
                    self.root.ids.play_button.disabled = False
                if hasattr(self.root.ids, 'pause_button'):
                    self.root.ids.pause_button.text = "Pause"
                    self.root.ids.pause_button.disabled = True
                
                # Show analysis results after playback completes
                show_pending_analysis_results(self)
                
                return False  # Stop scheduling this check
                
            return True  # Continue checking if still playing
                
        except Exception as e:
            print(f"Error in check_audio_ended: {e}")
            traceback.print_exc()
            
        return True  # Continue checking if still playing

    def update_playback_position(self, dt):
        """Update the current playback position"""
        try:
            if self.is_playing and self.controller:
                # Call controller to update its playback position
                self.controller.update_playback_position(dt)

                # For debug: occasionally print position
                if random.random() < 0.01:  # ~1% of frames
                    if hasattr(self.controller, 'playback_progress'):
                        progress = self.controller.playback_progress
                        print(f"App playback progress: {progress*100:.1f}%")
        except Exception as e:
            print(f"Error in app.update_playback_position: {e}")

    def update_waveform(self, dt):
        """Update the waveform plot"""
        if not hasattr(self, 'waveform_plot') or not self.waveform_plot:
            return

        try:
            self.waveform_plot.update_plot(dt)
        except Exception as e:
            print(f"Error updating waveform: {e}")

    def update_recording_waveform(self, dt):
        """Update the waveform during recording
        
        Args:
            dt (float): Delta time since last update
        """
        # Make sure we're recording
        if not self.is_recording:
            return False
            
        # Make sure we have a controller
        if not hasattr(self, 'controller') or not self.controller:
            return True
            
        # Get the latest waveform data from the controller
        if hasattr(self.controller, 'waveform_data') and self.controller.waveform_data is not None:
            # The controller is already updating the waveform_data directly
            # during recording, so we just need to ensure the plot is updated
            if hasattr(self, 'waveform_plot') and self.waveform_plot:
                # Print a debug message occasionally 
                if random.random() < 0.01:  # 1% chance to log
                    waveform_length = len(self.controller.waveform_data)
                    print(f"Recording waveform update: {waveform_length} samples")
            
            # Continue updating
            return True

    def update_file_label(self, filename=None, mode=None):
        """Update the file label with current file info

        Args:
            filename (str, optional): Path to the file. Defaults to None.
            mode (str, optional): Mode to display ('playing', 'saved', 'loaded'). Defaults to None.
        """
        if not filename:
            # Reset to default
            self.root.ids.file_label.text = "Audio Recorder Ready"
            return

        # Extract just the filename without the path
        base_filename = os.path.basename(filename)

        # Determine the appropriate message based on mode
        if mode == 'saved':
            self.root.ids.file_label.text = f"Saved: {base_filename}"
        elif mode == 'loaded':
            self.root.ids.file_label.text = f"Loaded: {base_filename}"
        elif mode == 'playing':
            self.root.ids.file_label.text = f"Playing: {base_filename}"
        elif self.is_playing:
            # If mode not specified but we're playing, show as playing
            self.root.ids.file_label.text = f"Playing: {base_filename}"
        else:
            # Default behavior for no specific mode
            self.root.ids.file_label.text = f"File: {base_filename}"

    def show_message(self, message):
        """Show a message dialog"""
        dialog = MDDialog(
            text=message,
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: dialog.dismiss()
                ),
            ],
        )
        dialog.open()

    def on_stop(self):
        """Events to run when the application stops"""
        # Stop any ongoing recording
        if self.is_recording:
            self.stop_recording()

        # Clean up audio resources
        self.controller.cleanup()

        # Clean up file managers
        if hasattr(self, 'file_manager'):
            self.file_manager.close()

        if hasattr(self, 'save_dir_file_manager'):
            self.save_dir_file_manager.close()

    def choose_save_directory(self):
        """Open dialog to choose save directory"""
        if self.is_mobile:
            try:
                from plyer import filechooser
                filechooser.choose_dir(on_selection=self.handle_mobile_dir_selection)
            except Exception as e:
                print(f"Mobile directory chooser error: {e}")
                # Fallback for testing
                self.show_message("Directory selection not available on this device")
                # Use default directory for mobile
                if platform == 'android':
                    default_dir = '/sdcard/Download'
                elif platform == 'ios':
                    default_dir = os.path.expanduser('~/Documents')
                else:
                    default_dir = os.path.expanduser('~')

                if hasattr(self.save_dialog.ids, 'directory_label'):
                    self.save_dialog.ids.directory_label.text = default_dir
                self.save_dialog.selected_directory = default_dir
        else:
            # Desktop file manager
            documents_path = os.path.join(os.path.expanduser("~"), "Documents")
            self.file_manager.show(documents_path)

    def handle_mobile_dir_selection(self, selection):
        """Handle directory selection from mobile directory chooser"""
        if not selection:
            return

        selected_dir = selection[0]
        if hasattr(self.save_dialog.ids, 'directory_label'):
            self.save_dialog.ids.directory_label.text = selected_dir
        self.save_dialog.selected_directory = selected_dir

    def select_save_directory(self, path):
        """Callback when save directory is selected"""
        # Close the file manager
        self.exit_file_manager()

        # Update the selected directory in the dialog
        content = self._save_dialog_temp.content_cls
        content.selected_directory = path

        # Update the directory label
        content.ids.directory_label.text = self._shorten_path(path)

        # Reopen the save dialog
        self.save_dialog = self._save_dialog_temp
        self.save_dialog.open()

    def _shorten_path(self, path):
        """Shorten a path for display"""
        home = os.path.expanduser("~")
        if path.startswith(home):
            return "~" + path[len(home):]
        return path
    
    def analyze_heart_sound(self):
        """Analyze heart sound and update UI with results"""
        if not hasattr(self, 'controller') or not self.controller:
            self.show_message("Audio controller not initialized")
            return False
            
        # Check if we have audio data to analyze
        if not hasattr(self.controller, 'waveform_data') or len(self.controller.waveform_data) == 0:
            self.show_message("No audio data available for analysis")
            return False
            
        try:
            # Analyze heart sound using controller
            result = self.controller.analyze_heart_sound()
            
            if not result:
                self.show_message("Analysis failed. Please try again.")
                return False
                
            # Update UI with results
            self.analysis_result = result['prediction'].capitalize()
            self.analysis_confidence = result['confidence'] * 100  # Convert to percentage
            
            # Set color based on prediction
            if result['prediction'] == 'normal':
                self.analysis_color = "#00AA00"  # Green for normal
            elif result['prediction'] == 'abnormal':
                self.analysis_color = "#AA0000"  # Red for abnormal
            else:
                self.analysis_color = "#888888"  # Gray for unclassified
                
            # Show message with result
            confidence_str = f"{self.analysis_confidence:.1f}%"
            self.show_message(f"Heart sound analysis: {self.analysis_result} (Confidence: {confidence_str})")
            
            return True
        except Exception as e:
            print(f"Error in heart sound analysis: {e}")
            traceback.print_exc()
            self.show_message(f"Error in analysis: {str(e)}")
            return False
    
    def generate_explanation(self):
        """Generate and display SHAP explanation for heart sound analysis"""
        if not hasattr(self, 'controller') or not self.controller:
            self.show_message("Audio controller not initialized")
            return False
            
        try:
            # Get explanation from controller
            explanation = self.controller.generate_explanation()
            
            if not explanation or not explanation.get('explanation_values'):
                self.show_message("Could not generate explanation. Please try again.")
                return False
                
            # Create a SHAP explanation plot
            if hasattr(self.controller.heart_sound_classifier, 'plot_explanation'):
                fig = self.controller.heart_sound_classifier.plot_explanation(
                    audio_data=self.controller.waveform_data,
                    fs=self.controller.sample_rate
                )
                
                # Create temporary file for the plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(os.path.expanduser("~"), "Documents", f"heart_explanation_{timestamp}.png")
                fig.savefig(plot_path)
                
                # TODO: Display the explanation plot in the app
                # For now, we'll just inform the user where the plot was saved
                self.show_message(f"Explanation plot saved to: {self._shorten_path(plot_path)}")
                plt.close(fig)  # Close the figure to free memory
                
            return True
        except Exception as e:
            print(f"Error generating explanation: {e}")
            traceback.print_exc()
            self.show_message(f"Error generating explanation: {str(e)}")
            return False

if __name__ == '__main__':
    AudioRecorderApp().run()
