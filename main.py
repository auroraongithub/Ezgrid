import sys
import json
import math
import subprocess
import shutil
import webbrowser
import re
import tempfile
import os
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from enum import Enum

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QPushButton, QListWidget, QListWidgetItem,
    QScrollArea, QLabel, QSpinBox, QDoubleSpinBox, QFormLayout, QFileDialog, QMenu,
    QSplitter, QFrame, QMessageBox, QAbstractItemView, QSizePolicy,
    QComboBox, QCheckBox, QGroupBox, QColorDialog, QProgressDialog, QDialog,
    QProgressBar, QPlainTextEdit, QToolButton, QTextEdit
)
from PySide6.QtCore import (
    Qt, Signal, QMimeData, QByteArray, QDataStream, QIODevice, QSize, QRectF, QPointF,
    QThread, QObject, QTimer, QProcess, QElapsedTimer
)
from PySide6.QtGui import (
    QAction, QDrag, QColor, QPalette, QPainter, QPen, QBrush, QFont, QDesktopServices,
    QPixmap, QImage
)


# =============================================================================
# FFMPEG/FFPROBE DISCOVERY
# =============================================================================

def find_ffmpeg() -> Optional[str]:
    """
    Find FFmpeg executable, checking bundled location first, then system PATH.
    
    Search Order:
    1. Bundled ffmpeg in app directory (ffmpeg.exe on Windows, ffmpeg on Unix)
    2. System PATH (using shutil.which)
    3. Direct test of common command (handles venv PATH issues)
    
    Returns:
        Path to ffmpeg executable, or None if not found
    """
    # Get the application directory
    app_dir = Path(__file__).parent.resolve()
    
    # Check for bundled ffmpeg
    if sys.platform == "win32":
        bundled_path = app_dir / "ffmpeg.exe"
    else:
        bundled_path = app_dir / "ffmpeg"
    
    if bundled_path.exists() and bundled_path.is_file():
        return str(bundled_path)
    
    # Try shutil.which first
    which_result = shutil.which("ffmpeg")
    if which_result:
        return which_result
    
    # Fallback: try to run ffmpeg directly to see if it's in PATH
    # This handles cases where shutil.which fails in venvs
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        if result.returncode == 0:
            return "ffmpeg"  # It's in PATH, command works
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    return None


def find_ffprobe() -> Optional[str]:
    """
    Find ffprobe executable, checking bundled location first, then system PATH.
    
    Returns:
        Path to ffprobe executable, or None if not found
    """
    app_dir = Path(__file__).parent.resolve()
    
    if sys.platform == "win32":
        bundled_path = app_dir / "ffprobe.exe"
    else:
        bundled_path = app_dir / "ffprobe"
    
    if bundled_path.exists() and bundled_path.is_file():
        return str(bundled_path)
    
    # Try shutil.which first
    which_result = shutil.which("ffprobe")
    if which_result:
        return which_result
    
    # Fallback: try to run ffprobe directly
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        if result.returncode == 0:
            return "ffprobe"  # It's in PATH, command works
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    return None


# =============================================================================
# ENUMS
# =============================================================================

class ScalingMode(Enum):
    """Video scaling modes for grid cells."""
    LETTERBOX = "letterbox"  # Fit, centered with bars
    CROP = "crop"            # Fill, centered with crop
    STRETCH = "stretch"      # Stretch to fill


class AnchorPosition(Enum):
    """Canvas anchor positions for grid placement."""
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class ResolutionPreset(Enum):
    """Output resolution presets."""
    HD_1080P = "1080p"
    QHD_1440P = "1440p"
    UHD_4K = "4k"
    CUSTOM = "custom"


class QualityPreset(Enum):
    """Export speed presets."""
    BEST_QUALITY = "best"     # slow preset, CRF 17
    BALANCED = "balanced"      # medium preset, CRF 19 (default)
    FAST_EXPORT = "fast"       # veryfast preset, CRF 25


class DurationMode(Enum):
    """Duration mode for export."""
    SHORTEST_CLIP = "shortest"
    CUSTOM = "custom"


class AudioSource(Enum):
    """Audio source for export."""
    NONE = "none"
    FIRST_CLIP = "first"
    SELECTED_CLIP = "selected"


# =============================================================================
# FFPROBE UTILITY (must be defined before Clip class)
# =============================================================================

def probe_video(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Use ffprobe to extract video metadata in JSON format.
    
    Extracts:
        - duration: Video duration in seconds
        - width: Video width in pixels
        - height: Video height in pixels
        - fps: Frame rate (as float)
        - has_audio: Whether the file has an audio stream
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Dictionary with metadata, or None if ffprobe fails
    """
    # Check if ffprobe is available (bundled or system PATH)
    ffprobe_path = find_ffprobe()
    if not ffprobe_path:
        return None
    
    try:
        # Run ffprobe with JSON output for streams and format
        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,  # Timeout after 10 seconds
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        
        # Extract format duration
        duration = float(data.get("format", {}).get("duration", 0))
        
        # Find video stream
        width = 0
        height = 0
        fps = 0.0
        has_audio = False
        
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "")
            
            if codec_type == "video" and width == 0:
                width = stream.get("width", 0)
                height = stream.get("height", 0)
                
                # Parse frame rate from avg_frame_rate or r_frame_rate
                # Format is typically "30000/1001" or "30/1"
                fps_str = stream.get("avg_frame_rate") or stream.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    if int(den) > 0:
                        fps = float(num) / float(den)
                else:
                    fps = float(fps_str) if fps_str else 0.0
                    
            elif codec_type == "audio":
                has_audio = True
        
        return {
            "duration": duration,
            "width": width,
            "height": height,
            "fps": round(fps, 3),  # Round to 3 decimal places
            "has_audio": has_audio
        }
        
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError):
        return None


def generate_thumbnail(video_path: str, size: Tuple[int, int] = (320, 180)) -> str:
    """
    Generate a thumbnail image from a video file using FFmpeg.
    
    Extracts a frame from 1 second into the video (or first frame if shorter)
    and saves it as a temporary JPEG file.
    
    Args:
        video_path: Path to the video file
        size: Thumbnail size (width, height), default 320x180 (16:9)
        
    Returns:
        Path to the generated thumbnail, or empty string if failed
    """
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return ""
    
    try:
        # Create a temporary file for the thumbnail
        # Use a persistent temp directory so thumbnails survive during session
        thumb_dir = Path(tempfile.gettempdir()) / "ezgrid_thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        
        # Create unique filename based on video path hash
        video_hash = abs(hash(video_path)) % (10 ** 10)
        thumb_path = thumb_dir / f"thumb_{video_hash}.jpg"
        
        # If thumbnail already exists, return it
        if thumb_path.exists():
            return str(thumb_path)
        
        # Generate thumbnail using FFmpeg
        # -ss 1: seek to 1 second (fast for most videos)
        # -vframes 1: extract only 1 frame
        # -vf scale: scale to target size maintaining aspect ratio
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite
            "-ss", "1",  # Seek to 1 second
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2:black",
            "-q:v", "2",  # High quality JPEG
            str(thumb_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        if result.returncode == 0 and thumb_path.exists():
            return str(thumb_path)
        
        # If seeking to 1s failed (video too short), try first frame
        cmd[3] = "0"  # Change -ss to 0
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        if result.returncode == 0 and thumb_path.exists():
            return str(thumb_path)
            
    except (subprocess.TimeoutExpired, OSError):
        pass
    
    return ""


# Global thumbnail cache for QPixmap objects
_thumbnail_cache: Dict[str, QPixmap] = {}


def get_thumbnail_pixmap(thumbnail_path: str) -> Optional[QPixmap]:
    """
    Get a QPixmap for a thumbnail, using cache for performance.
    
    Args:
        thumbnail_path: Path to thumbnail image file
        
    Returns:
        QPixmap or None if loading failed
    """
    if not thumbnail_path or not Path(thumbnail_path).exists():
        return None
    
    if thumbnail_path not in _thumbnail_cache:
        pixmap = QPixmap(thumbnail_path)
        if not pixmap.isNull():
            _thumbnail_cache[thumbnail_path] = pixmap
        else:
            return None
    
    return _thumbnail_cache.get(thumbnail_path)


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class Clip:
    """Represents an imported video clip with metadata."""
    path: str
    display_name: str
    duration: float = 0.0      # Duration in seconds
    width: int = 0             # Video width in pixels
    height: int = 0            # Video height in pixels
    fps: float = 0.0           # Frame rate
    has_audio: bool = False    # Whether clip has audio stream
    thumbnail_path: str = ""   # Path to generated thumbnail image
    
    @classmethod
    def from_path(cls, file_path: str) -> "Clip":
        """Create a Clip from a file path with metadata from ffprobe."""
        clip = cls(
            path=file_path,
            display_name=Path(file_path).name
        )
        # Try to probe metadata (non-blocking, fails gracefully)
        metadata = probe_video(file_path)
        if metadata:
            clip.duration = metadata.get("duration", 0.0)
            clip.width = metadata.get("width", 0)
            clip.height = metadata.get("height", 0)
            clip.fps = metadata.get("fps", 0.0)
            clip.has_audio = metadata.get("has_audio", False)
        
        # Generate thumbnail
        clip.thumbnail_path = generate_thumbnail(file_path)
        
        return clip


@dataclass
class GridSettings:
    """Grid configuration settings with full layout controls."""
    rows: int = 2
    columns: int = 2
    tile_width: int = 320      # Uniform tile width in pixels
    tile_height: int = 180     # Uniform tile height (16:9 = 320x180)
    gap_horizontal: int = 4    # Horizontal gap between cells
    gap_vertical: int = 4      # Vertical gap between cells
    background_color: str = "#000000"  # Background color (hex)
    scaling_mode: str = "letterbox"    # ScalingMode value
    anchor: str = "center"             # AnchorPosition value
    center_last_row: bool = True       # Center clips in last row if partial
    canvas_width: int = 1920           # Output canvas width
    canvas_height: int = 1080          # Output canvas height (16:9)
    auto_tile_size: bool = True        # Auto-compute tile size to fit canvas


@dataclass
class ExportSettings:
    """Export configuration settings."""
    # Resolution
    resolution_preset: str = "1080p"  # ResolutionPreset value
    custom_width: int = 1920
    custom_height: int = 1080
    
    # FPS
    fps_auto: bool = True
    fps_manual: float = 30.0
    
    # Quality/Speed
    quality_preset: str = "balanced"  # QualityPreset value (best/balanced/fast)
    
    # Performance
    use_proxies: bool = True  # Use proxy files for faster export
    
    # Duration
    duration_mode: str = "shortest"  # DurationMode value
    custom_duration: float = 10.0  # seconds
    
    # Audio
    audio_source: str = "none"  # AudioSource value
    audio_clip_index: int = 0  # For "selected" audio source
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the actual output resolution."""
        preset = ResolutionPreset(self.resolution_preset)
        if preset == ResolutionPreset.HD_1080P:
            return (1920, 1080)
        elif preset == ResolutionPreset.QHD_1440P:
            return (2560, 1440)
        elif preset == ResolutionPreset.UHD_4K:
            return (3840, 2160)
        else:  # CUSTOM
            return (self.custom_width, self.custom_height)


@dataclass
class CellLayout:
    """Computed layout for a single grid cell."""
    row: int
    col: int
    x: float           # X position on canvas
    y: float           # Y position on canvas
    width: float       # Cell width
    height: float      # Cell height
    clip_index: int    # Index of clip in this cell (-1 if empty)
    is_visible: bool   # Whether this cell should be rendered


@dataclass 
class GridLayout:
    """Computed layout for the entire grid."""
    cells: List[CellLayout]
    total_width: float      # Total grid width (without offset)
    total_height: float     # Total grid height (without offset)
    offset_x: float         # X offset for centering/anchoring
    offset_y: float         # Y offset for centering/anchoring
    scale: float            # Scale factor for preview


# =============================================================================
# UNDO/REDO SYSTEM
# =============================================================================

class UndoCommand:
    """Base class for undoable commands."""
    
    def __init__(self, description: str):
        self.description = description
    
    def undo(self):
        """Undo this command."""
        raise NotImplementedError
    
    def redo(self):
        """Redo this command."""
        raise NotImplementedError


class ReorderClipsCommand(UndoCommand):
    """Command for reordering clips (move from_index to to_index)."""
    
    def __init__(self, model: "DataModel", from_index: int, to_index: int):
        super().__init__(f"Reorder clip {from_index} to {to_index}")
        self.model = model
        self.from_index = from_index
        self.to_index = to_index
    
    def undo(self):
        """Move back from to_index to from_index."""
        clip = self.model.clips.pop(self.to_index)
        self.model.clips.insert(self.from_index, clip)
    
    def redo(self):
        """Move from from_index to to_index."""
        clip = self.model.clips.pop(self.from_index)
        self.model.clips.insert(self.to_index, clip)


class SwapClipsCommand(UndoCommand):
    """Command for swapping two clips."""
    
    def __init__(self, model: "DataModel", index_a: int, index_b: int):
        super().__init__(f"Swap clips {index_a} and {index_b}")
        self.model = model
        self.index_a = index_a
        self.index_b = index_b
    
    def undo(self):
        """Swap back."""
        self.model.clips[self.index_a], self.model.clips[self.index_b] = \
            self.model.clips[self.index_b], self.model.clips[self.index_a]
    
    def redo(self):
        """Swap again."""
        self.model.clips[self.index_a], self.model.clips[self.index_b] = \
            self.model.clips[self.index_b], self.model.clips[self.index_a]


class RemoveClipCommand(UndoCommand):
    """Command for removing a clip."""
    
    def __init__(self, model: "DataModel", clip_index: int, clip: "Clip"):
        super().__init__(f"Remove clip {clip_index}")
        self.model = model
        self.clip_index = clip_index
        self.clip = clip  # Store the clip for undo
    
    def undo(self):
        """Re-insert the clip."""
        self.model.clips.insert(self.clip_index, self.clip)
    
    def redo(self):
        """Remove the clip again."""
        self.model.clips.pop(self.clip_index)


class UndoStack:
    """Manages undo/redo command stack."""
    
    def __init__(self, max_size: int = 100):
        self.undo_stack: List[UndoCommand] = []
        self.redo_stack: List[UndoCommand] = []
        self.max_size = max_size
    
    def push(self, command: UndoCommand):
        """Push a command onto the undo stack and execute it."""
        command.redo()  # Execute the command
        self.undo_stack.append(command)
        self.redo_stack.clear()  # Clear redo stack on new action
        
        # Limit stack size
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
    
    def undo(self) -> bool:
        """Undo the last command. Returns True if successful."""
        if not self.undo_stack:
            return False
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        return True
    
    def redo(self) -> bool:
        """Redo the last undone command. Returns True if successful."""
        if not self.redo_stack:
            return False
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)
        return True
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.redo_stack) > 0
    
    def clear(self):
        """Clear all undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()


# =============================================================================
# LOGGING AND DETAILS PANEL SYSTEM
# =============================================================================

@dataclass
class ErrorEntry:
    """Represents an error entry for the error history."""
    timestamp: str
    summary: str
    details: str
    exit_code: int = 0
    output_path: str = ""
    command: str = ""


class LogManager(QObject):
    """
    Singleton manager for logging, progress tracking, and error history.
    
    Signals:
        log_added: Emitted when a log line is added (message)
        progress_updated: Emitted when progress changes (task, current, total, item_name, elapsed_ms, eta_ms)
        error_added: Emitted when an error is recorded (ErrorEntry)
        ffmpeg_stats_updated: Emitted with parsed FFmpeg stats (time_str, fps_str, speed_str)
    """
    
    log_added = Signal(str)
    progress_updated = Signal(str, int, int, str, int, int)  # task, current, total, item, elapsed_ms, eta_ms
    error_added = Signal(object)  # ErrorEntry
    ffmpeg_stats_updated = Signal(str, str, str)  # time, fps, speed
    task_started = Signal(str)  # task name
    task_finished = Signal(str, bool)  # task name, success
    
    _instance = None
    
    @classmethod
    def instance(cls) -> 'LogManager':
        """Get the singleton LogManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True
        self.errors: List[ErrorEntry] = []
        self._log_buffer: List[str] = []
        self._flush_timer = QTimer()
        self._flush_timer.timeout.connect(self._flush_logs)
        self._flush_timer.setInterval(150)  # Batch logs every 150ms
    
    def log(self, message: str, timestamp: bool = True):
        """Add a log message (buffered for efficiency)."""
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {message}"
        else:
            line = message
        self._log_buffer.append(line)
        if not self._flush_timer.isActive():
            self._flush_timer.start()
    
    def _flush_logs(self):
        """Flush buffered log messages to UI."""
        if self._log_buffer:
            combined = "\n".join(self._log_buffer)
            self.log_added.emit(combined)
            self._log_buffer.clear()
        self._flush_timer.stop()
    
    def flush_now(self):
        """Force immediate flush of log buffer."""
        self._flush_logs()
    
    def update_progress(self, task: str, current: int, total: int, item_name: str = "", 
                        elapsed_ms: int = 0, eta_ms: int = 0):
        """Update progress for a task."""
        self.progress_updated.emit(task, current, total, item_name, elapsed_ms, eta_ms)
    
    def update_ffmpeg_stats(self, time_str: str, fps_str: str, speed_str: str):
        """Update FFmpeg-specific stats."""
        self.ffmpeg_stats_updated.emit(time_str, fps_str, speed_str)
    
    def start_task(self, task_name: str):
        """Signal that a task has started."""
        self.log(f"Started: {task_name}")
        self.task_started.emit(task_name)
    
    def finish_task(self, task_name: str, success: bool = True):
        """Signal that a task has finished."""
        status = "completed" if success else "failed"
        self.log(f"Finished: {task_name} ({status})")
        self.flush_now()
        self.task_finished.emit(task_name, success)
    
    def add_error(self, summary: str, details: str, exit_code: int = 0, 
                  output_path: str = "", command: str = ""):
        """Add an error to the error history."""
        entry = ErrorEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary=summary,
            details=details,
            exit_code=exit_code,
            output_path=output_path,
            command=command
        )
        self.errors.append(entry)
        self.error_added.emit(entry)
        self.log(f"ERROR: {summary}")
    
    def clear_errors(self):
        """Clear error history."""
        self.errors.clear()


# Global log manager instance
def get_log_manager() -> LogManager:
    """Get the global LogManager instance."""
    return LogManager()


class DetailsPanel(QWidget):
    """
    Collapsible details panel with Progress, Logs, and Errors tabs.
    
    Shows detailed information about long-running tasks, live logs,
    and error history.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._collapsed = True
        self._elapsed_timer = QElapsedTimer()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with toggle button
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background-color: #2a2a2a; border-top: 1px solid #404040;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        
        self.toggle_btn = QToolButton()
        self.toggle_btn.setArrowType(Qt.RightArrow)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet("QToolButton { border: none; }")
        self.toggle_btn.clicked.connect(self._toggle_collapsed)
        header_layout.addWidget(self.toggle_btn)
        
        self.header_label = QLabel("Details")
        self.header_label.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        header_layout.addWidget(self.header_label)
        
        header_layout.addStretch()
        
        # Status indicator in header
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #808080; font-size: 11px;")
        header_layout.addWidget(self.status_label)
        
        layout.addWidget(header)
        
        # Content area (collapsible)
        self.content = QWidget()
        self.content.setVisible(False)
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setFixedHeight(200)
        
        # Progress tab
        self._setup_progress_tab()
        
        # Logs tab
        self._setup_logs_tab()
        
        # Errors tab
        self._setup_errors_tab()
        
        content_layout.addWidget(self.tabs)
        layout.addWidget(self.content)
        
        # Connect to log manager
        log_mgr = get_log_manager()
        log_mgr.log_added.connect(self._on_log_added)
        log_mgr.progress_updated.connect(self._on_progress_updated)
        log_mgr.error_added.connect(self._on_error_added)
        log_mgr.ffmpeg_stats_updated.connect(self._on_ffmpeg_stats)
        log_mgr.task_started.connect(self._on_task_started)
        log_mgr.task_finished.connect(self._on_task_finished)
    
    def _setup_progress_tab(self):
        """Setup the Progress tab."""
        progress_widget = QWidget()
        layout = QVBoxLayout(progress_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Task name
        self.task_label = QLabel("No active task")
        self.task_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.task_label)
        
        # Progress bar
        self.detail_progress = QProgressBar()
        self.detail_progress.setRange(0, 100)
        self.detail_progress.setValue(0)
        layout.addWidget(self.detail_progress)
        
        # Current item
        self.item_label = QLabel("")
        self.item_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.item_label)
        
        # Time info (elapsed / ETA)
        time_layout = QHBoxLayout()
        self.elapsed_label = QLabel("Elapsed: --")
        self.elapsed_label.setStyleSheet("color: #808080;")
        time_layout.addWidget(self.elapsed_label)
        
        self.eta_label = QLabel("ETA: --")
        self.eta_label.setStyleSheet("color: #808080;")
        time_layout.addWidget(self.eta_label)
        time_layout.addStretch()
        layout.addLayout(time_layout)
        
        # FFmpeg stats (only shown during export)
        self.ffmpeg_stats_label = QLabel("")
        self.ffmpeg_stats_label.setStyleSheet("color: #808080; font-family: monospace;")
        self.ffmpeg_stats_label.setVisible(False)
        layout.addWidget(self.ffmpeg_stats_label)
        
        layout.addStretch()
        self.tabs.addTab(progress_widget, "Progress")
    
    def _setup_logs_tab(self):
        """Setup the Logs tab."""
        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                font-family: Consolas, monospace;
                font-size: 11px;
                border: none;
            }
        """)
        self.logs_text.setMaximumBlockCount(1000)  # Limit to 1000 lines
        self.tabs.addTab(self.logs_text, "Logs")
    
    def _setup_errors_tab(self):
        """Setup the Errors tab."""
        errors_widget = QWidget()
        layout = QVBoxLayout(errors_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Error list
        self.error_list = QListWidget()
        self.error_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                color: #ff6666;
                border: none;
            }
            QListWidget::item:selected {
                background-color: #3a3a3a;
            }
        """)
        self.error_list.itemClicked.connect(self._on_error_clicked)
        layout.addWidget(self.error_list)
        
        # Error details (expandable)
        self.error_details = QTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.setVisible(False)
        self.error_details.setMaximumHeight(100)
        self.error_details.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                font-family: Consolas, monospace;
                font-size: 10px;
                border: 1px solid #404040;
            }
        """)
        layout.addWidget(self.error_details)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.copy_cmd_btn = QPushButton("Copy Command")
        self.copy_cmd_btn.setVisible(False)
        self.copy_cmd_btn.clicked.connect(self._copy_error_command)
        btn_layout.addWidget(self.copy_cmd_btn)
        
        clear_btn = QPushButton("Clear Errors")
        clear_btn.clicked.connect(self._clear_errors)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.tabs.addTab(errors_widget, "Errors (0)")
    
    def _toggle_collapsed(self):
        """Toggle the collapsed state."""
        self._collapsed = not self._collapsed
        self.content.setVisible(not self._collapsed)
        self.toggle_btn.setArrowType(Qt.DownArrow if not self._collapsed else Qt.RightArrow)
    
    def expand(self):
        """Expand the panel."""
        if self._collapsed:
            self._toggle_collapsed()
            self.toggle_btn.setChecked(True)
    
    def _on_log_added(self, message: str):
        """Handle new log message."""
        self.logs_text.appendPlainText(message)
        # Auto-scroll to bottom
        scrollbar = self.logs_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_progress_updated(self, task: str, current: int, total: int, item: str, 
                             elapsed_ms: int, eta_ms: int):
        """Handle progress update."""
        self.task_label.setText(task)
        
        if total > 0:
            percent = int((current / total) * 100)
            self.detail_progress.setValue(percent)
            self.item_label.setText(f"Processing: {item}" if item else f"{current} / {total}")
        else:
            self.detail_progress.setValue(current)
            self.item_label.setText(item)
        
        # Format elapsed time
        if elapsed_ms > 0:
            elapsed_sec = elapsed_ms / 1000
            self.elapsed_label.setText(f"Elapsed: {self._format_time(elapsed_sec)}")
        
        # Format ETA
        if eta_ms > 0:
            eta_sec = eta_ms / 1000
            self.eta_label.setText(f"ETA: {self._format_time(eta_sec)}")
        else:
            self.eta_label.setText("ETA: --")
        
        # Update header status
        if total > 0:
            self.status_label.setText(f"{task}: {current}/{total}")
    
    def _on_ffmpeg_stats(self, time_str: str, fps_str: str, speed_str: str):
        """Handle FFmpeg stats update."""
        self.ffmpeg_stats_label.setVisible(True)
        parts = []
        if time_str:
            parts.append(f"time={time_str}")
        if fps_str:
            parts.append(f"fps={fps_str}")
        if speed_str:
            parts.append(f"speed={speed_str}")
        self.ffmpeg_stats_label.setText("  ".join(parts))
    
    def _on_task_started(self, task: str):
        """Handle task start."""
        self.task_label.setText(task)
        self.detail_progress.setValue(0)
        self.item_label.setText("")
        self.elapsed_label.setText("Elapsed: 0s")
        self.eta_label.setText("ETA: --")
        self.ffmpeg_stats_label.setVisible(False)
        self.status_label.setText(f"{task}...")
    
    def _on_task_finished(self, task: str, success: bool):
        """Handle task finish."""
        if success:
            self.detail_progress.setValue(100)
            self.status_label.setText(f"{task}: Done")
        else:
            self.status_label.setText(f"{task}: Failed")
            self.expand()
            self.tabs.setCurrentIndex(2)  # Switch to Errors tab
    
    def _on_error_added(self, entry: ErrorEntry):
        """Handle new error."""
        self.error_list.addItem(f"[{entry.timestamp}] {entry.summary}")
        self._update_error_tab_title()
        self.expand()
        self.tabs.setCurrentIndex(2)
    
    def _on_error_clicked(self, item):
        """Handle error list item click."""
        idx = self.error_list.row(item)
        log_mgr = get_log_manager()
        if 0 <= idx < len(log_mgr.errors):
            entry = log_mgr.errors[idx]
            self.error_details.setVisible(True)
            self.error_details.setText(
                f"Exit Code: {entry.exit_code}\n"
                f"Output: {entry.output_path}\n\n"
                f"Details:\n{entry.details}"
            )
            self.copy_cmd_btn.setVisible(bool(entry.command))
            self._current_error_cmd = entry.command
    
    def _copy_error_command(self):
        """Copy the error command to clipboard."""
        if hasattr(self, '_current_error_cmd') and self._current_error_cmd:
            QApplication.clipboard().setText(self._current_error_cmd)
    
    def _clear_errors(self):
        """Clear error history."""
        get_log_manager().clear_errors()
        self.error_list.clear()
        self.error_details.setVisible(False)
        self.copy_cmd_btn.setVisible(False)
        self._update_error_tab_title()
    
    def _update_error_tab_title(self):
        """Update the Errors tab title with count."""
        count = len(get_log_manager().errors)
        self.tabs.setTabText(2, f"Errors ({count})")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 0:
            return "--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"


class ErrorDetailDialog(QDialog):
    """Dialog showing error details with expandable info and copy command button."""
    
    def __init__(self, summary: str, details: str, command: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Error")
        self.setMinimumWidth(450)
        self.command = command
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Summary
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("font-weight: bold; color: #ff6666;")
        layout.addWidget(summary_label)
        
        # Show details toggle
        self.details_visible = False
        self.toggle_btn = QPushButton("Show Details ▼")
        self.toggle_btn.clicked.connect(self._toggle_details)
        layout.addWidget(self.toggle_btn)
        
        # Details area (hidden by default)
        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlainText(details)
        self.details_text.setVisible(False)
        self.details_text.setMaximumHeight(200)
        self.details_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                font-family: Consolas, monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.details_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        if command:
            copy_btn = QPushButton("Copy Command")
            copy_btn.clicked.connect(self._copy_command)
            btn_layout.addWidget(copy_btn)
        
        btn_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
    
    def _toggle_details(self):
        """Toggle details visibility."""
        self.details_visible = not self.details_visible
        self.details_text.setVisible(self.details_visible)
        self.toggle_btn.setText("Hide Details ▲" if self.details_visible else "Show Details ▼")
        self.adjustSize()
    
    def _copy_command(self):
        """Copy command to clipboard."""
        QApplication.clipboard().setText(self.command)


@dataclass
class DataModel:
    """
    Central data model for the application.
    
    Attributes:
        clips: List of imported Clip objects
        grid_settings: Current grid configuration
        export_settings: Current export configuration
        selected_clip_index: Currently selected clip in import list (-1 if none)
        selected_cell_index: Currently selected grid cell (-1 if none)
    """
    clips: list = field(default_factory=list)
    grid_settings: GridSettings = field(default_factory=GridSettings)
    export_settings: ExportSettings = field(default_factory=ExportSettings)
    selected_clip_index: int = -1
    selected_cell_index: int = -1
    
    def add_clip(self, clip: Clip) -> int:
        """Add a clip and return its index. Auto-adjusts grid if needed."""
        self.clips.append(clip)
        self._auto_adjust_grid()
        return len(self.clips) - 1
    
    def remove_clip(self, clip_index: int):
        """Remove a clip and update selection."""
        if 0 <= clip_index < len(self.clips):
            self.clips.pop(clip_index)
            # Update selection
            if self.selected_clip_index == clip_index:
                self.selected_clip_index = -1
            elif self.selected_clip_index > clip_index:
                self.selected_clip_index -= 1
            self._auto_adjust_grid()
    
    def reorder_clips(self, from_index: int, to_index: int):
        """Reorder clips in the list (grid follows import order)."""
        if from_index == to_index:
            return
        if not (0 <= from_index < len(self.clips) and 0 <= to_index < len(self.clips)):
            return
        clip = self.clips.pop(from_index)
        self.clips.insert(to_index, clip)
    
    def swap_clips(self, index_a: int, index_b: int):
        """Swap two clips in the list."""
        if 0 <= index_a < len(self.clips) and 0 <= index_b < len(self.clips):
            self.clips[index_a], self.clips[index_b] = self.clips[index_b], self.clips[index_a]
    
    def _auto_adjust_grid(self):
        """
        Auto-adjust grid rows/columns based on clip count.
        Uses smallest grid that fits all clips, preferring near-square layouts.
        """
        n = len(self.clips)
        if n == 0:
            self.grid_settings.rows = 2
            self.grid_settings.columns = 2
            return
        
        # Find optimal near-square grid that fits n clips
        # Start with ceil(sqrt(n)) as the ideal columns
        ideal_cols = math.ceil(math.sqrt(n))
        
        # Try a range of column counts around the ideal
        best_rows, best_cols = 1, n  # Worst case fallback
        best_score = float('inf')
        
        # Search range: from ideal-2 to ideal+2, but ensure reasonable bounds
        min_cols = max(1, ideal_cols - 2)
        max_cols = min(n, ideal_cols + 2)
        
        for cols in range(min_cols, max_cols + 1):
            rows = math.ceil(n / cols)
            empty_cells = rows * cols - n
            
            # Score based on:
            # 1. How square the layout is (minimize aspect ratio difference)
            # 2. How many empty cells (minimize waste)
            aspect_diff = abs(rows - cols)
            score = aspect_diff * 10 + empty_cells
            
            if score < best_score:
                best_score = score
                best_rows = rows
                best_cols = cols
        
        self.grid_settings.rows = best_rows
        self.grid_settings.columns = best_cols
        
        # Also auto-compute tile size if enabled
        if self.grid_settings.auto_tile_size:
            self._auto_compute_tile_size()
    
    def _auto_compute_tile_size(self):
        """
        Auto-compute tile_width and tile_height so the entire grid fits 
        inside the canvas (16:9) including gaps. Tiles remain strictly 16:9.
        
        Uses the formula from fix.md to ensure proper scaling:
        - max_tile_w_by_w = floor((out_w - (cols-1)*gap_x) / cols)
        - max_tile_h_by_h = floor((out_h - (rows-1)*gap_y) / rows)
        - max_tile_w_by_h = floor(max_tile_h_by_h * 16 / 9)
        - tile_w = min(max_tile_w_by_w, max_tile_w_by_h)
        - tile_h = floor(tile_w * 9 / 16)
        - Force both to even integers >= 2
        """
        gs = self.grid_settings
        rows = gs.rows
        cols = gs.columns
        gap_h = gs.gap_horizontal
        gap_v = gs.gap_vertical
        
        # Use export resolution as canvas size (single source of truth)
        canvas_w, canvas_h = self.export_settings.get_resolution()
        
        if rows <= 0 or cols <= 0:
            return
        
        # Max tile width constrained by canvas width
        # out_w = cols * tile_w + (cols - 1) * gap_h
        # tile_w = (out_w - (cols - 1) * gap_h) / cols
        max_tile_w_by_w = (canvas_w - max(0, cols - 1) * gap_h) // cols
        
        # Max tile height constrained by canvas height
        # out_h = rows * tile_h + (rows - 1) * gap_v
        # tile_h = (out_h - (rows - 1) * gap_v) / rows
        max_tile_h_by_h = (canvas_h - max(0, rows - 1) * gap_v) // rows
        
        # Max tile width if constrained by height (maintaining 16:9)
        max_tile_w_by_h = int(max_tile_h_by_h * 16 / 9)
        
        # Use the smaller constraint to ensure both width and height fit
        tile_w = min(max_tile_w_by_w, max_tile_w_by_h)
        tile_h = int(tile_w * 9 / 16)
        
        # Force even integers >= 2 (required for video encoding)
        tile_w = max(2, tile_w)
        tile_h = max(2, tile_h)
        if tile_w % 2 != 0:
            tile_w -= 1
        if tile_h % 2 != 0:
            tile_h -= 1
        
        gs.tile_width = tile_w
        gs.tile_height = tile_h
        
        # Update canvas dimensions to match export resolution
        gs.canvas_width = canvas_w
        gs.canvas_height = canvas_h
        
        # Debug logging
        print(f"[AUTO TILE] Resolution: {canvas_w}x{canvas_h}, Grid: {rows}x{cols}, "
              f"Computed tiles: {tile_w}x{tile_h}")
    
    def compute_grid_layout(self, preview_width: float, preview_height: float) -> GridLayout:
        """
        Compute the full grid layout with centering math.
        
        This calculates:
        1. Cell positions based on tile size and gaps
        2. Last row centering if enabled
        3. Global offset based on anchor position
        4. Scale factor for preview display
        
        Args:
            preview_width: Width of preview area in pixels
            preview_height: Height of preview area in pixels
            
        Returns:
            GridLayout with all computed cell positions
        """
        gs = self.grid_settings
        n_clips = len(self.clips)
        rows = gs.rows
        cols = gs.columns
        
        # Cell dimensions (strictly 16:9)
        cell_w = gs.tile_width
        cell_h = gs.tile_height
        gap_h = gs.gap_horizontal
        gap_v = gs.gap_vertical
        
        # =====================================================================
        # GRID TOTAL SIZE CALCULATION
        # Total width = cols * cell_w + (cols - 1) * gap_h
        # Total height = rows * cell_h + (rows - 1) * gap_v
        # =====================================================================
        total_grid_w = cols * cell_w + max(0, cols - 1) * gap_h
        total_grid_h = rows * cell_h + max(0, rows - 1) * gap_v
        
        # =====================================================================
        # SCALE FACTOR FOR PREVIEW
        # Scale to fit inside preview area while maintaining aspect ratio
        # Canvas aspect ratio is maintained (16:9 by default)
        # =====================================================================
        canvas_w = gs.canvas_width
        canvas_h = gs.canvas_height
        
        # Scale canvas to fit preview
        scale_x = preview_width / canvas_w if canvas_w > 0 else 1.0
        scale_y = preview_height / canvas_h if canvas_h > 0 else 1.0
        scale = min(scale_x, scale_y)
        
        # =====================================================================
        # ANCHOR/CENTERING OFFSET CALCULATION
        # Computes global dx, dy to position the grid on the canvas
        # =====================================================================
        anchor = AnchorPosition(gs.anchor)
        
        # Available space on canvas
        available_w = canvas_w
        available_h = canvas_h
        
        # Calculate offset based on anchor
        if anchor == AnchorPosition.CENTER:
            offset_x = (available_w - total_grid_w) / 2
            offset_y = (available_h - total_grid_h) / 2
        elif anchor == AnchorPosition.TOP_LEFT:
            offset_x = 0
            offset_y = 0
        elif anchor == AnchorPosition.TOP_RIGHT:
            offset_x = available_w - total_grid_w
            offset_y = 0
        elif anchor == AnchorPosition.BOTTOM_LEFT:
            offset_x = 0
            offset_y = available_h - total_grid_h
        elif anchor == AnchorPosition.BOTTOM_RIGHT:
            offset_x = available_w - total_grid_w
            offset_y = available_h - total_grid_h
        else:
            offset_x = (available_w - total_grid_w) / 2
            offset_y = (available_h - total_grid_h) / 2
        
        # =====================================================================
        # CELL POSITION CALCULATION
        # Row-major order, with optional last row centering
        # =====================================================================
        cells = []
        
        # Determine how many clips are in the last row
        clips_in_last_row = n_clips % cols if n_clips % cols != 0 else cols
        last_row = (n_clips - 1) // cols if n_clips > 0 else 0
        
        for row in range(rows):
            for col in range(cols):
                cell_index = row * cols + col
                clip_index = cell_index if cell_index < n_clips else -1
                
                # =====================================================================
                # BASE POSITION CALCULATION
                # x = col * (cell_w + gap_h)
                # y = row * (cell_h + gap_v)
                # =====================================================================
                base_x = col * (cell_w + gap_h)
                base_y = row * (cell_h + gap_v)
                
                # =====================================================================
                # LAST ROW CENTERING
                # If center_last_row is enabled and this is the last row with clips:
                # Offset = (cols - clips_in_last_row) * (cell_w + gap_h) / 2
                # =====================================================================
                last_row_offset_x = 0
                if gs.center_last_row and row == last_row and clips_in_last_row < cols and n_clips > 0:
                    # Only offset cells that have clips
                    if clip_index >= 0:
                        last_row_offset_x = (cols - clips_in_last_row) * (cell_w + gap_h) / 2
                
                # Final position (before global offset)
                cell_x = base_x + last_row_offset_x
                cell_y = base_y
                
                cells.append(CellLayout(
                    row=row,
                    col=col,
                    x=cell_x,
                    y=cell_y,
                    width=cell_w,
                    height=cell_h,
                    clip_index=clip_index,
                    is_visible=(clip_index >= 0)
                ))
        
        return GridLayout(
            cells=cells,
            total_width=total_grid_w,
            total_height=total_grid_h,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale
        )
    
    def get_clip_at_cell(self, cell_index: int) -> int:
        """Get clip index at a cell position (row-major order). Returns -1 if empty."""
        if 0 <= cell_index < len(self.clips):
            return cell_index
        return -1
    
    def get_cell_for_clip(self, clip_index: int) -> int:
        """Get cell index for a clip. Returns clip_index since grid follows import order."""
        if 0 <= clip_index < len(self.clips):
            return clip_index
        return -1
    
    def clear(self):
        """Reset the data model to initial state."""
        self.clips.clear()
        self.grid_settings = GridSettings()
        self.selected_clip_index = -1
        self.selected_cell_index = -1
    
    def to_dict(self) -> dict:
        """Serialize model to dictionary for saving."""
        return {
            "clips": [asdict(c) for c in self.clips],
            "grid_settings": asdict(self.grid_settings),
            "export_settings": asdict(self.export_settings),
            "selected_clip_index": self.selected_clip_index,
            "selected_cell_index": self.selected_cell_index
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DataModel":
        """Deserialize model from dictionary."""
        model = cls()
        model.clips = [Clip(**c) for c in data.get("clips", [])]
        gs = data.get("grid_settings", {})
        model.grid_settings = GridSettings(**gs)
        es = data.get("export_settings", {})
        model.export_settings = ExportSettings(**es)
        model.selected_clip_index = data.get("selected_clip_index", -1)
        model.selected_cell_index = data.get("selected_cell_index", -1)
        return model


# =============================================================================
# FFMPEG UTILITIES
# =============================================================================

def get_quality_params(quality_preset: str) -> Tuple[str, int]:
    """
    Map export speed preset to FFmpeg encoding parameters.
    
    Export Speed Presets:
        - best: Slow encoding, best quality (preset=slow, CRF 17)
        - balanced: Medium speed, good quality/size (preset=medium, CRF 19)
        - fast: Very fast encoding, lower quality (preset=veryfast, CRF 25)
    
    Args:
        quality_preset: One of "best", "balanced", "fast"
        
    Returns:
        Tuple of (preset_name, crf_value) for x264 encoding
    """
    presets = {
        "best": ("slow", 17),
        "balanced": ("medium", 19),
        "fast": ("veryfast", 25),
        # Legacy support
        "high": ("slow", 17),
        "small": ("veryfast", 25),
    }
    return presets.get(quality_preset, ("medium", 19))


# =============================================================================
# PROXY FILE MANAGEMENT
# =============================================================================

def get_proxy_cache_dir() -> Path:
    """Get or create the proxy cache directory."""
    cache_dir = Path(__file__).parent / ".proxy_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def compute_proxy_config_key(model: 'DataModel') -> str:
    """
    Compute a stable cache key for proxy configuration.
    
    This key represents the complete export configuration that affects
    how proxies should be generated. When any of these parameters change,
    proxies must be regenerated.
    
    Key includes:
    - Export resolution (out_w, out_h)
    - Grid layout (rows, cols, tile_w, tile_h)
    - Gaps (gap_h, gap_v)
    - Scaling mode (letterbox/crop/stretch)
    - Duration settings (mode + custom duration)
    - FPS settings (auto/manual)
    
    Returns:
        Hash string for config subdirectory
    """
    import hashlib
    
    gs = model.grid_settings
    es = model.export_settings
    
    # Get current export resolution
    out_w, out_h = es.get_resolution()
    
    # Get current tile dimensions (after auto-sizing)
    tile_w = gs.tile_width
    tile_h = gs.tile_height
    
    # Build config key string with all relevant parameters
    config_parts = [
        f"res={out_w}x{out_h}",
        f"grid={gs.rows}x{gs.columns}",
        f"tile={tile_w}x{tile_h}",
        f"gap={gs.gap_horizontal}x{gs.gap_vertical}",
        f"scale={gs.scaling_mode}",
        f"dur={es.duration_mode}:{es.custom_duration}",
        f"fps={es.fps_auto}:{es.fps_manual}",
    ]
    
    config_str = "|".join(config_parts)
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def compute_proxy_clip_key(clip: Clip) -> str:
    """
    Compute a stable cache key for a specific clip file.
    
    Key based on:
    - Original file path
    - File modification time
    
    Returns:
        Hash string suitable for filename
    """
    import hashlib
    
    # Get file mtime
    try:
        mtime = os.path.getmtime(clip.path)
    except:
        mtime = 0
    
    # Create key from file identity
    key_str = f"{clip.path}|{mtime}"
    hash_obj = hashlib.sha256(key_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def get_proxy_cache_subdir(model: 'DataModel') -> Path:
    """
    Get the cache subdirectory for the current configuration.
    
    Each unique export configuration gets its own subdirectory to avoid
    conflicts when settings change.
    """
    cache_root = get_proxy_cache_dir()
    config_key = compute_proxy_config_key(model)
    subdir = cache_root / config_key
    subdir.mkdir(exist_ok=True)
    return subdir


def get_proxy_path(model: 'DataModel', clip: Clip) -> Path:
    """Get the path where proxy file should be stored for current config."""
    cache_subdir = get_proxy_cache_subdir(model)
    clip_key = compute_proxy_clip_key(clip)
    return cache_subdir / f"proxy_{clip_key}.mp4"


def proxy_exists(model: 'DataModel', clip: Clip) -> bool:
    """Check if valid proxy file exists for this clip with current config."""
    proxy_path = get_proxy_path(model, clip)
    return proxy_path.exists() and proxy_path.is_file()


def get_proxy_params(model: 'DataModel') -> Tuple[int, int, float]:
    """
    Get proxy generation parameters from current model state.
    
    Returns:
        Tuple of (proxy_w, proxy_h, duration)
    """
    gs = model.grid_settings
    es = model.export_settings
    
    # Proxy dimensions based on CURRENT tile size (after auto-sizing)
    tile_w = gs.tile_width if gs.tile_width % 2 == 0 else gs.tile_width + 1
    tile_h = gs.tile_height if gs.tile_height % 2 == 0 else gs.tile_height + 1
    
    # Ensure proxy is at least 160x90 (standard low-res minimum)
    proxy_w = max(tile_w, 160)
    proxy_h = max(tile_h, 90)
    
    # Compute duration for proxies
    duration = compute_output_duration(model.clips, es)
    
    return proxy_w, proxy_h, duration


def generate_proxy_command(clip: Clip, proxy_w: int, proxy_h: int, duration: float, scaling_mode: str, output_path: str) -> List[str]:
    """
    Generate FFmpeg command to create a proxy file.
    
    Proxy specs:
    - Scale to proxy resolution (maintain aspect based on scaling_mode)
    - Fast encoding (veryfast preset, CRF 28)
    - Trim to specified duration
    - Copy audio if present
    
    Args:
        clip: Source clip
        proxy_w: Proxy width (even)
        proxy_h: Proxy height (even)
        duration: Duration to encode
        scaling_mode: "letterbox", "crop", or "stretch"
        output_path: Output file path
        
    Returns:
        FFmpeg command as list
    """
    ffmpeg_path = find_ffmpeg() or "ffmpeg"
    
    cmd = [ffmpeg_path, "-y", "-hide_banner"]
    
    # Input file
    cmd.extend(["-i", clip.path])
    
    # Trim duration
    if duration > 0:
        cmd.extend(["-t", str(duration)])
    
    # Video filter: scale to proxy size based on scaling mode
    mode = ScalingMode(scaling_mode)
    if mode == ScalingMode.LETTERBOX:
        # Scale to fit, then pad
        vf = f"scale={proxy_w}:{proxy_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={proxy_w}:{proxy_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    elif mode == ScalingMode.CROP:
        # Scale to fill, then crop
        vf = f"scale={proxy_w}:{proxy_h}:force_original_aspect_ratio=increase:force_divisible_by=2,crop={proxy_w}:{proxy_h},setsar=1"
    else:  # STRETCH
        # Scale directly
        vf = f"scale={proxy_w}:{proxy_h}:force_divisible_by=2,setsar=1"
    
    cmd.extend(["-vf", vf])
    
    # Fast encoding for proxy
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p"
    ])
    
    # Copy audio if present (or encode quickly)
    if clip.has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    
    cmd.append(output_path)
    
    return cmd


def compute_output_duration(clips: List[Clip], export_settings: 'ExportSettings') -> float:
    """
    Compute the final output duration based on clips and export settings.
    
    Duration Modes:
        - shortest: Use the shortest clip's duration
        - custom: Use custom duration (clips will loop if shorter, trim if longer)
    
    Args:
        clips: List of Clip objects with duration metadata
        export_settings: Export configuration
        
    Returns:
        Output duration in seconds (0 if no valid clips)
    """
    if not clips:
        return 0.0
    
    # Find shortest clip duration (ignoring clips with 0 duration)
    valid_durations = [c.duration for c in clips if c.duration > 0]
    if not valid_durations:
        return 0.0
    
    shortest = min(valid_durations)
    
    mode = DurationMode(export_settings.duration_mode)
    if mode == DurationMode.SHORTEST_CLIP:
        return shortest
    else:  # CUSTOM
        # Return custom duration (clips will loop if needed)
        return export_settings.custom_duration


def build_scale_filter(input_idx: int, clip: Clip, tile_width: int, tile_height: int,
                       scaling_mode: str, output_duration: float = 0) -> str:
    """
    Build FFmpeg scale filter for a single clip.
    
    Handles three scaling modes:
        - letterbox: Scale to fit, pad to fill (black bars)
        - crop: Scale to fill, crop excess
        - stretch: Scale to exact size (may distort)
    
    If output_duration > clip.duration, adds looping filters to extend the clip.
    
    All outputs have SAR=1 (square pixels) and even dimensions.
    
    Args:
        input_idx: Input stream index (0-based)
        clip: Clip object with source dimensions
        tile_width: Target tile width (must be even)
        tile_height: Target tile height (must be even)
        scaling_mode: One of "letterbox", "crop", "stretch"
        output_duration: Target output duration (0 = no looping)
        
    Returns:
        FFmpeg filter string for this input
    """
    # Ensure tile dimensions are even (required for most video codecs)
    tile_w = tile_width if tile_width % 2 == 0 else tile_width + 1
    tile_h = tile_height if tile_height % 2 == 0 else tile_height + 1
    
    # Check if we need to loop this clip
    needs_loop = output_duration > 0 and clip.duration > 0 and clip.duration < output_duration
    
    # Calculate how many loops we need
    if needs_loop:
        # loop filter: loop=loop=N loops the video N times (total N+1 playbacks)
        # We want ceil(output_duration / clip.duration) total playbacks
        import math
        total_playbacks = math.ceil(output_duration / clip.duration)
        loop_count = max(0, total_playbacks - 1)  # -1 because loop=0 means play once
    
    mode = ScalingMode(scaling_mode)
    
    if mode == ScalingMode.LETTERBOX:
        # LETTERBOX (fit): Scale to fit inside tile, preserving aspect ratio
        if needs_loop:
            return (
                f"[{input_idx}:v]"
                f"loop=loop={loop_count}:size=32767:start=0,"
                f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1"
                f"[v{input_idx}]"
            )
        else:
            return (
                f"[{input_idx}:v]"
                f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1"
                f"[v{input_idx}]"
            )
    elif mode == ScalingMode.CROP:
        # CROP (fill): Scale to fill tile (overflow), then crop to exact size
        if needs_loop:
            return (
                f"[{input_idx}:v]"
                f"loop=loop={loop_count}:size=32767:start=0,"
                f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={tile_w}:{tile_h},"
                f"setsar=1"
                f"[v{input_idx}]"
            )
        else:
            return (
                f"[{input_idx}:v]"
                f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={tile_w}:{tile_h},"
                f"setsar=1"
                f"[v{input_idx}]"
            )
    else:  # STRETCH
        # STRETCH: Scale to exact size (may distort aspect ratio)
        if needs_loop:
            return (
                f"[{input_idx}:v]"
                f"loop=loop={loop_count}:size=32767:start=0,"
                f"scale={tile_w}:{tile_h}:force_divisible_by=2,"
                f"setsar=1"
                f"[v{input_idx}]"
            )
        else:
            return (
                f"[{input_idx}:v]"
                f"scale={tile_w}:{tile_h}:force_divisible_by=2,"
                f"setsar=1"
                f"[v{input_idx}]"
            )


def build_filter_complex(model: 'DataModel') -> str:
    """
    Build the complete FFmpeg filter_complex string for video grid composition.
    
    Filter Graph Structure:
    ======================
    1. Create background canvas: color filter with canvas size
    2. Per-input scaling: Scale each input to uniform tile size with SAR=1
    3. xstack: Combine all scaled inputs into a grid
    4. overlay: Place grid onto background canvas at offset position
    
    This approach avoids pad filter issues by using overlay instead.
    
    xstack Layout String Format:
    ===========================
    - Format: "x0_y0|x1_y1|x2_y2|..."
    - Each position is relative to the stack origin
    - Positions include gaps between cells
    
    Canvas Anchoring:
    =================
    - The grid may not fill the entire canvas
    - Global offset (dx, dy) positions the grid on the canvas
    - Implemented using overlay filter positioning
    
    Args:
        model: DataModel with clips, grid_settings, and export_settings
        
    Returns:
        Complete filter_complex string
    """
    clips = model.clips
    gs = model.grid_settings
    es = model.export_settings
    
    if not clips:
        return ""
    
    n_clips = len(clips)
    
    # Ensure tile dimensions are even (required for video codecs)
    tile_w = gs.tile_width if gs.tile_width % 2 == 0 else gs.tile_width + 1
    tile_h = gs.tile_height if gs.tile_height % 2 == 0 else gs.tile_height + 1
    
    gap_h = gs.gap_horizontal
    gap_v = gs.gap_vertical
    rows = gs.rows
    cols = gs.columns
    
    # Get canvas dimensions from export settings (ensure even)
    canvas_w, canvas_h = es.get_resolution()
    canvas_w = canvas_w if canvas_w % 2 == 0 else canvas_w + 1
    canvas_h = canvas_h if canvas_h % 2 == 0 else canvas_h + 1
    
    # Parse background color (remove # if present)
    bg_color = gs.background_color.lstrip("#")
    
    filters = []
    
    # =========================================================================
    # STEP 1: Create background canvas using color filter
    # =========================================================================
    # color=c=COLOR:s=WIDTHxHEIGHT:d=DURATION[bg]
    # We use a long duration; it will be trimmed by shortest=1 in overlay
    bg_filter = f"color=c=#{bg_color}:s={canvas_w}x{canvas_h}:d=3600[bg]"
    filters.append(bg_filter)
    
    # =========================================================================
    # STEP 2: Scale each input to uniform tile size (with looping if needed)
    # =========================================================================
    output_duration = compute_output_duration(clips, es)
    for i, clip in enumerate(clips):
        scale_filter = build_scale_filter(i, clip, tile_w, tile_h, gs.scaling_mode, output_duration)
        filters.append(scale_filter)
    
    # =========================================================================
    # STEP 3: Build xstack layout string
    # =========================================================================
    # Compute cell positions (same logic as compute_grid_layout)
    clips_in_last_row = n_clips % cols if n_clips % cols != 0 else cols
    last_row = (n_clips - 1) // cols if n_clips > 0 else 0
    
    # Calculate total grid size
    total_grid_w = cols * tile_w + max(0, cols - 1) * gap_h
    total_grid_h = rows * tile_h + max(0, rows - 1) * gap_v
    
    # Calculate anchor offset
    anchor = AnchorPosition(gs.anchor)
    if anchor == AnchorPosition.CENTER:
        offset_x = (canvas_w - total_grid_w) // 2
        offset_y = (canvas_h - total_grid_h) // 2
    elif anchor == AnchorPosition.TOP_LEFT:
        offset_x = 0
        offset_y = 0
    elif anchor == AnchorPosition.TOP_RIGHT:
        offset_x = canvas_w - total_grid_w
        offset_y = 0
    elif anchor == AnchorPosition.BOTTOM_LEFT:
        offset_x = 0
        offset_y = canvas_h - total_grid_h
    elif anchor == AnchorPosition.BOTTOM_RIGHT:
        offset_x = canvas_w - total_grid_w
        offset_y = canvas_h - total_grid_h
    else:
        offset_x = (canvas_w - total_grid_w) // 2
        offset_y = (canvas_h - total_grid_h) // 2
    
    # Ensure offsets are non-negative (grid larger than canvas clips to edge)
    offset_x = max(0, offset_x)
    offset_y = max(0, offset_y)
    
    # Build xstack positions
    xstack_positions = []
    for i in range(n_clips):
        row = i // cols
        col = i % cols
        
        # Base position
        base_x = col * (tile_w + gap_h)
        base_y = row * (tile_h + gap_v)
        
        # Last row centering offset
        last_row_offset_x = 0
        if gs.center_last_row and row == last_row and clips_in_last_row < cols:
            last_row_offset_x = (cols - clips_in_last_row) * (tile_w + gap_h) // 2
        
        cell_x = base_x + last_row_offset_x
        cell_y = base_y
        
        xstack_positions.append(f"{cell_x}_{cell_y}")
    
    # xstack input labels
    xstack_inputs = "".join(f"[v{i}]" for i in range(n_clips))
    xstack_layout = "|".join(xstack_positions)
    
    # xstack filter with background color fill for gaps
    xstack_filter = f"{xstack_inputs}xstack=inputs={n_clips}:layout={xstack_layout}:fill=#{bg_color}:shortest=0[grid]"
    filters.append(xstack_filter)
    
    # =========================================================================
    # STEP 4: Overlay grid onto background canvas
    # =========================================================================
    # This avoids pad filter issues - overlay always works
    # [bg][grid]overlay=x=OFFSET_X:y=OFFSET_Y:shortest=1[vout]
    overlay_filter = f"[bg][grid]overlay=x={offset_x}:y={offset_y}:shortest=1[vout]"
    filters.append(overlay_filter)
    
    # =========================================================================
    # STEP 5: Handle audio (if needed)
    # =========================================================================
    audio_source = AudioSource(es.audio_source)
    if audio_source != AudioSource.NONE:
        # Determine which clip provides audio
        audio_clip_idx = 0  # Default to first clip
        if audio_source == AudioSource.SELECTED_CLIP:
            audio_clip_idx = es.audio_clip_index
        
        # Check if the audio clip exists and has audio
        if 0 <= audio_clip_idx < len(clips) and clips[audio_clip_idx].has_audio:
            audio_clip = clips[audio_clip_idx]
            
            # Check if we need to loop the audio
            if output_duration > 0 and audio_clip.duration > 0 and audio_clip.duration < output_duration:
                # Loop audio using aloop filter
                # aloop=loop=N:size=SAMPLES (N=-1 means infinite loop)
                # We'll use -1 and rely on -t to trim
                audio_filter = f"[{audio_clip_idx}:a]aloop=loop=-1:size=2e9[aout]"
                filters.append(audio_filter)
            else:
                # No looping needed, just pass through
                audio_filter = f"[{audio_clip_idx}:a]acopy[aout]"
                filters.append(audio_filter)
    
    return ";".join(filters)


def build_ffmpeg_command(model: 'DataModel', output_path: str, proxy_paths: Optional[List[str]] = None) -> Tuple[List[str], Optional[str]]:
    """
    Generate the complete FFmpeg command as a Python list.
    
    Command Structure:
    ==================
    ffmpeg [global_options] [input_files] [filter_complex] [output_options] output_file
    
    Global Options:
    ---------------
    -y: Overwrite output without asking
    -hide_banner: Suppress FFmpeg banner
    
    Input Files:
    ------------
    -i file1.mp4 -i file2.mp4 ...
    Uses proxy files if proxy_paths is provided and valid
    
    Filter Complex:
    ---------------
    For many clips (>50), writes filter to temp file and uses -filter_complex_script
    Otherwise uses -filter_complex "..." (from build_filter_complex)
    
    Output Options:
    ---------------
    -map "[vout]": Use the final video output from filter
    -map "N:a"?: Optional audio mapping
    -c:v libx264: Video codec (H.264)
    -preset X: Encoding preset from quality settings
    -crf Y: Constant Rate Factor from quality settings
    -pix_fmt yuv420p: Pixel format for compatibility
    -t duration: Output duration
    -r fps: Output frame rate (if not auto)
    
    Args:
        model: DataModel with all settings
        output_path: Path for output video file
        proxy_paths: Optional list of proxy file paths (one per clip, empty string if no proxy)
        
    Returns:
        Tuple of (FFmpeg command as list, filter_script_path or None)
    """
    clips = model.clips
    es = model.export_settings
    
    if not clips:
        return [], None
    
    # Find ffmpeg (bundled or system PATH)
    ffmpeg_path = find_ffmpeg() or "ffmpeg"
    
    cmd = [ffmpeg_path]
    
    # =========================================================================
    # GLOBAL OPTIONS
    # =========================================================================
    cmd.extend(["-y", "-hide_banner"])
    
    # =========================================================================
    # INPUT FILES
    # =========================================================================
    # Use proxy files if available, otherwise use original clips
    use_proxies = proxy_paths and len(proxy_paths) == len(clips)
    
    for i, clip in enumerate(clips):
        if use_proxies and proxy_paths[i]:
            # Use proxy file
            cmd.extend(["-i", proxy_paths[i]])
        else:
            # Use original file
            cmd.extend(["-i", clip.path])
    
    # =========================================================================
    # FILTER COMPLEX
    # For many clips, use filter_complex_script to avoid command line length limits
    # =========================================================================
    filter_complex = build_filter_complex(model)
    filter_script_path = None
    
    # Use script file if filter is long or many inputs (avoids Windows cmd line limit)
    if len(clips) > 50 or len(filter_complex) > 4000:
        # Write filter to temp file
        import tempfile
        fd, filter_script_path = tempfile.mkstemp(suffix=".txt", text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(filter_complex)
            cmd.extend(["-filter_complex_script", filter_script_path])
        except:
            if filter_script_path and os.path.exists(filter_script_path):
                os.unlink(filter_script_path)
            filter_script_path = None
            # Fallback to inline
            cmd.extend(["-filter_complex", filter_complex])
    else:
        cmd.extend(["-filter_complex", filter_complex])
    
    # =========================================================================
    # OUTPUT MAPPING
    # =========================================================================
    # Map video output from filter
    cmd.extend(["-map", "[vout]"])
    
    # Audio mapping based on settings
    audio_source = AudioSource(es.audio_source)
    if audio_source != AudioSource.NONE:
        # Determine which clip provides audio
        audio_clip_idx = 0  # Default to first clip
        if audio_source == AudioSource.SELECTED_CLIP:
            audio_clip_idx = es.audio_clip_index
        
        # Check if the audio clip exists and has audio
        if 0 <= audio_clip_idx < len(clips) and clips[audio_clip_idx].has_audio:
            # Audio is processed in filter_complex and output as [aout]
            cmd.extend(["-map", "[aout]"])
    # AudioSource.NONE: no audio mapping
    
    # =========================================================================
    # VIDEO CODEC AND QUALITY
    # =========================================================================
    preset, crf = get_quality_params(es.quality_preset)
    cmd.extend([
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p"  # For maximum compatibility
    ])
    
    # =========================================================================
    # AUDIO CODEC (if audio is included)
    # =========================================================================
    if audio_source != AudioSource.NONE:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    
    # =========================================================================
    # FRAME RATE
    # =========================================================================
    if es.fps_auto:
        # Use first clip's fps if available
        if clips[0].fps > 0:
            cmd.extend(["-r", str(clips[0].fps)])
        # If no fps info, let FFmpeg use default
    else:
        cmd.extend(["-r", str(es.fps_manual)])
    
    # =========================================================================
    # DURATION
    # =========================================================================
    output_duration = compute_output_duration(clips, es)
    if output_duration > 0:
        cmd.extend(["-t", str(output_duration)])
    
    # =========================================================================
    # OUTPUT FILE
    # =========================================================================
    cmd.append(output_path)
    
    return cmd, filter_script_path


# =============================================================================
# FFMPEG AVAILABILITY DIALOG
# =============================================================================

class FFmpegNotFoundDialog(QDialog):
    """
    Dialog shown when FFmpeg is not found.
    
    Provides options to:
    - Download FFmpeg (opens browser)
    - Retry detection
    - Cancel
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FFmpeg Required")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Message
        message = QLabel(
            "FFmpeg is required for video export but was not found.\n\n"
            "FFmpeg is a free, open-source tool for video processing.\n"
            "Please download and install it, then click Retry.\n\n"
            "Installation options:\n"
            "• Windows: Download from gyan.dev or ffmpeg.org\n"
            "• Add ffmpeg to your system PATH, or\n"
            "• Place ffmpeg.exe in the same folder as this app"
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("Download FFmpeg")
        self.download_btn.clicked.connect(self._on_download)
        btn_layout.addWidget(self.download_btn)
        
        self.retry_btn = QPushButton("Retry")
        self.retry_btn.clicked.connect(self._on_retry)
        btn_layout.addWidget(self.retry_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
        self._ffmpeg_found = False
    
    def _on_download(self):
        """Open FFmpeg download page in browser."""
        webbrowser.open("https://ffmpeg.org/download.html")
    
    def _on_retry(self):
        """Re-check for FFmpeg availability."""
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path:
            self._ffmpeg_found = True
            self.accept()
        else:
            QMessageBox.warning(
                self, "Not Found",
                "FFmpeg still not found.\n\n"
                "Make sure ffmpeg is in your PATH or\n"
                "placed in the application folder."
            )
    
    def was_ffmpeg_found(self) -> bool:
        """Return True if FFmpeg was found after retry."""
        return self._ffmpeg_found


# =============================================================================
# FFMPEG EXPORT WORKER (Background Thread)
# =============================================================================

class FFmpegWorker(QObject):
    """
    Worker object that runs FFmpeg in a background thread.
    
    Parses FFmpeg stderr output to extract progress information.
    FFmpeg outputs progress in the format: time=HH:MM:SS.MS
    
    Signals:
        progress_updated: Emitted with (current_time_seconds, percentage)
        finished: Emitted when FFmpeg completes (success: bool, message: str)
        error: Emitted on error with message
    """
    
    progress_updated = Signal(float, int)  # (time_seconds, percentage)
    finished = Signal(bool, str)  # (success, message)
    error = Signal(str)
    
    def __init__(self, cmd: List[str], total_duration: float, filter_script_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.total_duration = total_duration
        self.filter_script_path = filter_script_path
        self.process: Optional[subprocess.Popen] = None
        self._cancelled = False
        self._stderr_output = []  # Store stderr for error reporting
    
    def run(self):
        """
        Execute FFmpeg command and parse progress from stderr.
        
        FFmpeg outputs progress info to stderr in format:
        frame=  123 fps= 30 q=28.0 size=    1234kB time=00:00:04.10 bitrate=2467.8kbits/s
        
        We parse the time= field to calculate progress percentage.
        """
        # Note: Cannot use LogManager from worker thread (causes Qt timer warnings)
        # All logging is done via stderr capture instead
        
        try:
            # Start FFmpeg process
            # Use CREATE_NO_WINDOW on Windows to hide console
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Regex to parse time from FFmpeg output
            # Matches: time=00:01:23.45 or time=01:23.45
            time_pattern = re.compile(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})')
            
            # Read stderr line by line for progress updates
            # FFmpeg writes progress to stderr
            while True:
                if self._cancelled:
                    break
                
                line = self.process.stderr.readline()
                if not line:
                    # Check if process has finished
                    if self.process.poll() is not None:
                        break
                    continue
                
                # Store stderr line for error reporting
                self._stderr_output.append(line)
                
                # Parse time= from the line
                match = time_pattern.search(line)
                if match:
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = int(match.group(3))
                    centiseconds = int(match.group(4))
                    
                    current_time = hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
                    
                    # Calculate percentage
                    if self.total_duration > 0:
                        percentage = min(100, int((current_time / self.total_duration) * 100))
                    else:
                        percentage = 0
                    
                    self.progress_updated.emit(current_time, percentage)
            
            # Wait for process to complete
            return_code = self.process.wait()
            
            if self._cancelled:
                self.finished.emit(False, "Export cancelled by user.")
            elif return_code == 0:
                self.finished.emit(True, "Export completed successfully!")
            else:
                # Build detailed error message
                # Get last 50 lines of stderr for context
                error_lines = self._stderr_output[-50:] if len(self._stderr_output) > 50 else self._stderr_output
                stderr_text = ''.join(error_lines)
                
                # Common error interpretations
                error_hints = []
                if 'Padded dimensions cannot be smaller' in stderr_text:
                    error_hints.append("\n\nPossible cause: Grid size calculation error.")
                    error_hints.append("Try adjusting canvas size or grid layout.")
                elif 'Invalid argument' in stderr_text or return_code == 4294967274 or return_code == -22:
                    error_hints.append("\n\nPossible cause: Invalid filter configuration or command line too long.")
                    error_hints.append("Try reducing the number of clips or grid complexity.")
                elif 'No such file' in stderr_text:
                    error_hints.append("\n\nPossible cause: Input file not found or invalid path.")
                elif 'Permission denied' in stderr_text:
                    error_hints.append("\n\nPossible cause: No write permission to output location.")
                
                error_msg = f"FFmpeg failed with exit code {return_code}\n\nLast stderr output:\n{stderr_text}{''.join(error_hints)}"
                self.finished.emit(False, error_msg)
                
        except Exception as e:
            self.error.emit(f"Error running FFmpeg: {str(e)}")
    
    def cancel(self):
        """
        Cancel the FFmpeg process.
        
        Sends SIGTERM (or terminates on Windows) to allow FFmpeg
        to clean up properly before exiting.
        """
        self._cancelled = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            # Give it a moment to clean up, then force kill if needed
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        # Clean up temp filter script if it exists
        if self.filter_script_path and os.path.exists(self.filter_script_path):
            try:
                os.unlink(self.filter_script_path)
            except:
                pass


# =============================================================================
# IMPORT WORKER (Background Thread)
# =============================================================================

class ImportWorker(QObject):
    """
    Worker object that imports video files in a background thread.
    
    Probes video metadata and generates thumbnails for each file,
    emitting progress updates and the resulting Clip objects.
    
    Signals:
        progress_updated: Emitted with (current_index, total_count, file_name)
        clip_ready: Emitted when a clip is fully loaded
        finished: Emitted when all files are processed
        error: Emitted on error with message
    """
    
    progress_updated = Signal(int, int, str)  # (current, total, filename)
    clip_ready = Signal(object)  # Clip object
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, file_paths: List[str], parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self._cancelled = False
    
    def run(self):
        """Process all files and emit clips as they're ready."""
        logger = LogManager.instance()
        total = len(self.file_paths)
        logger.log(f"Starting import of {total} file(s)...")
        
        for i, path in enumerate(self.file_paths):
            if self._cancelled:
                logger.warning("Import cancelled by user")
                break
            
            filename = Path(path).name
            self.progress_updated.emit(i + 1, total, filename)
            logger.log(f"Importing ({i + 1}/{total}): {filename}")
            
            try:
                # Create clip (probes metadata + generates thumbnail)
                clip = Clip.from_path(path)
                self.clip_ready.emit(clip)
                logger.log(f"  ✓ Loaded: {clip.width}x{clip.height}, {clip.duration:.1f}s, {clip.fps:.1f}fps")
            except Exception as e:
                error_msg = f"Error loading {filename}: {str(e)}"
                logger.add_error("Import Error", error_msg)
                self.error.emit(error_msg)
        
        logger.log(f"Import finished. Processed {total} file(s).")
        self.finished.emit()
    
    def cancel(self):
        """Cancel the import operation."""
        self._cancelled = True


# =============================================================================
# IMPORT PROGRESS DIALOG
# =============================================================================

class ImportProgressDialog(QDialog):
    """
    Progress dialog for importing video files.
    
    Shows:
    - Progress bar (0-100%)
    - "Importing X / N…" text
    - Current filename being processed
    - Cancel button
    
    The dialog runs import in a background thread to keep the GUI responsive.
    """
    
    # Signal emitted when clips are ready to be added to model
    clips_ready = Signal(list)  # List of Clip objects
    
    def __init__(self, file_paths: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Importing Videos...")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.file_paths = file_paths
        self.clips: List[Clip] = []
        self._cancelled = False
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Status label
        self.status_label = QLabel(f"Importing 0 / {len(file_paths)}…")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Current file label
        self.file_label = QLabel("")
        self.file_label.setStyleSheet("color: #808080; font-size: 11px;")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(file_paths))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn, alignment=Qt.AlignRight)
        
        # Setup worker and thread
        self.thread = QThread()
        self.worker = ImportWorker(file_paths)
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.clip_ready.connect(self._on_clip_ready)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        
        # Start processing
        self.thread.start()
    
    def _on_progress(self, current: int, total: int, filename: str):
        """Update progress display."""
        self.status_label.setText(f"Importing {current} / {total}…")
        self.file_label.setText(filename)
        self.progress_bar.setValue(current)
    
    def _on_clip_ready(self, clip: Clip):
        """Handle a clip being ready."""
        self.clips.append(clip)
    
    def _on_finished(self):
        """Handle import completion."""
        self.thread.quit()
        self.thread.wait()
        self.clips_ready.emit(self.clips)
        self.accept()
    
    def _on_error(self, message: str):
        """Handle error during import."""
        # Logged by worker, continue with other files
        pass
    
    def _on_cancel(self):
        """Handle cancel button."""
        self._cancelled = True
        self.worker.cancel()
        self.thread.quit()
        self.thread.wait()
        # Still emit whatever clips we managed to load
        self.clips_ready.emit(self.clips)
        self.reject()
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self.thread.isRunning():
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)


# =============================================================================
# PROXY GENERATION WORKER & DIALOG
# =============================================================================

class ProxyGenerationWorker(QObject):
    """
    Worker that generates proxy files for clips in a background thread.
    
    Signals:
        progress_updated: Emitted with (current_index, total_count, clip_name)
        proxy_ready: Emitted when a proxy is generated (clip_index, proxy_path)
        finished: Emitted when all proxies are processed
        error: Emitted on error with message
    """
    
    progress_updated = Signal(int, int, str)  # (current, total, clip_name)
    proxy_ready = Signal(int, str)  # (clip_index, proxy_path)
    finished = Signal(bool, str)  # (success, message)
    error = Signal(str)
    
    def __init__(self, model: 'DataModel', parent=None):
        super().__init__(parent)
        self.model = model
        self.clips = model.clips
        self._cancelled = False
    
    def run(self):
        """Generate proxies for all clips."""
        total = len(self.clips)
        generated = 0
        skipped = 0
        
        # Get proxy parameters from model (includes current tile size, duration, etc.)
        proxy_w, proxy_h, duration = get_proxy_params(self.model)
        scaling_mode = self.model.grid_settings.scaling_mode
        
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                self.finished.emit(False, "Proxy generation cancelled.")
                return
            
            clip_name = clip.display_name
            self.progress_updated.emit(i + 1, total, clip_name)
            
            # Check if proxy already exists with current config
            if proxy_exists(self.model, clip):
                proxy_path = str(get_proxy_path(self.model, clip))
                self.proxy_ready.emit(i, proxy_path)
                skipped += 1
                continue
            
            # Generate proxy
            try:
                proxy_path = str(get_proxy_path(self.model, clip))
                cmd = generate_proxy_command(clip, proxy_w, proxy_h, duration, scaling_mode, proxy_path)
                
                # Run FFmpeg to generate proxy
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                    timeout=300  # 5 minute timeout per proxy
                )
                
                if result.returncode == 0:
                    self.proxy_ready.emit(i, proxy_path)
                    generated += 1
                else:
                    error_msg = f"Proxy generation failed for {clip_name}: FFmpeg error {result.returncode}"
                    self.error.emit(error_msg)
            
            except Exception as e:
                error_msg = f"Error generating proxy for {clip_name}: {str(e)}"
                self.error.emit(error_msg)
        
        # Summary
        msg = f"Proxy generation complete. Generated: {generated}, Reused: {skipped}"
        self.finished.emit(True, msg)
    
    def cancel(self):
        """Cancel proxy generation."""
        self._cancelled = True


class ProxyGenerationDialog(QDialog):
    """
    Progress dialog for proxy file generation.
    
    Shows:
    - Progress bar
    - Current clip being processed
    - Cancel button
    """
    
    def __init__(self, model: DataModel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generating Proxy Files...")
        self.setModal(True)
        self.setMinimumWidth(450)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        clips = model.clips
        self.proxy_paths: List[str] = [""] * len(clips)  # Store proxy paths by clip index
        self._success = False
        self._message = ""
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Info label
        info_label = QLabel(
            "Generating proxy files for faster export.\n"
            "This only needs to be done once per clip and will be reused."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(clips))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel(f"Processing 0 / {len(clips)}...")
        layout.addWidget(self.status_label)
        
        # Current clip label
        self.clip_label = QLabel("")
        self.clip_label.setStyleSheet("color: #888;")
        layout.addWidget(self.clip_label)
        
        # Cancel button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # Create worker and thread
        self.thread = QThread()
        self.worker = ProxyGenerationWorker(model)
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.proxy_ready.connect(self._on_proxy_ready)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        
        # Start thread
        self.thread.start()
    
    def _on_progress(self, current: int, total: int, clip_name: str):
        """Update progress bar and labels."""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Processing {current} / {total}...")
        self.clip_label.setText(f"Current: {clip_name}")
    
    def _on_proxy_ready(self, clip_index: int, proxy_path: str):
        """Store proxy path for clip."""
        self.proxy_paths[clip_index] = proxy_path
    
    def _on_finished(self, success: bool, message: str):
        """Handle completion."""
        self._success = success
        self._message = message
        self.thread.quit()
        self.thread.wait()
        self.accept()
    
    def _on_error(self, message: str):
        """Handle error (logged but continue with other clips)."""
        pass
    
    def _on_cancel(self):
        """Cancel proxy generation."""
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
        self.worker.cancel()
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self.thread.isRunning():
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)
    
    def get_proxy_paths(self) -> List[str]:
        """Get the list of proxy paths (empty string if not generated)."""
        return self.proxy_paths
    
    def was_successful(self) -> bool:
        """Return True if all proxies were generated successfully."""
        return self._success
    
    def get_message(self) -> str:
        """Get completion message."""
        return self._message


# =============================================================================
# EXPORT PROGRESS DIALOG
# =============================================================================

class ExportProgressDialog(QDialog):
    """
    Progress dialog for FFmpeg export with cancel support.
    
    Shows:
    - Progress bar (0-100%)
    - Current time / Total time
    - Cancel button
    
    The dialog runs FFmpeg in a background thread to keep the GUI responsive.
    """
    
    def __init__(self, cmd: List[str], total_duration: float, output_path: str, filter_script_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting Video...")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.output_path = output_path
        self.total_duration = total_duration
        self.filter_script_path = filter_script_path
        self._success = False
        self._message = ""
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Output file label
        output_label = QLabel(f"Output: {Path(output_path).name}")
        output_label.setWordWrap(True)
        layout.addWidget(output_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Time label
        self.time_label = QLabel("Time: 0.0s / {:.1f}s".format(total_duration))
        layout.addWidget(self.time_label)
        
        # Status label
        self.status_label = QLabel("Starting export...")
        layout.addWidget(self.status_label)
        
        # Cancel button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # Create worker and thread
        self.thread = QThread()
        self.worker = FFmpegWorker(cmd, total_duration, filter_script_path)
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        
        # Start the thread
        self.thread.start()
    
    def _on_progress(self, current_time: float, percentage: int):
        """Update progress bar and time label."""
        self.progress_bar.setValue(percentage)
        self.time_label.setText(f"Time: {current_time:.1f}s / {self.total_duration:.1f}s")
        self.status_label.setText(f"Encoding... {percentage}%")
    
    def _on_finished(self, success: bool, message: str):
        """Handle FFmpeg completion."""
        self._success = success
        self._message = message
        self._cleanup_thread()
        
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("Complete!")
        
        self.accept()
    
    def _on_error(self, message: str):
        """Handle FFmpeg error."""
        self._success = False
        self._message = message
        self._cleanup_thread()
        self.reject()
    
    def _on_cancel(self):
        """Cancel the export."""
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
        self.worker.cancel()
    
    def _cleanup_thread(self):
        """Clean up the worker thread."""
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)  # Wait up to 3 seconds
        
        # Clean up temp filter script
        if self.filter_script_path and os.path.exists(self.filter_script_path):
            try:
                os.unlink(self.filter_script_path)
            except:
                pass
    
    def closeEvent(self, event):
        """Handle dialog close (X button)."""
        if self.thread.isRunning():
            self.worker.cancel()
            self._cleanup_thread()
        super().closeEvent(event)
    
    def was_successful(self) -> bool:
        """Return True if export completed successfully."""
        return self._success
    
    def get_message(self) -> str:
        """Return the completion/error message."""
        return self._message


# =============================================================================
# CANVAS GRID PREVIEW WIDGET
# =============================================================================

class CanvasPreview(QWidget):
    """
    Canvas-based grid preview that maintains aspect ratio and renders cells.
    
    The preview area represents the final export canvas.
    - Canvas maintains the selected output aspect ratio (default 16:9)
    - Background color applies only to the canvas area
    - Grid cells are strictly 16:9
    - Cells scale uniformly to fit inside the preview canvas
    - Supports drag-reorder of tiles within the grid with visual reflow
    """
    
    cell_clicked = Signal(int)  # Emits cell index
    clip_dropped = Signal(int, int)  # Emits (clip_index, cell_index) for swap
    clip_reordered = Signal(int, int)  # Emits (from_index, to_index) for move
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model: Optional[DataModel] = None
        self.layout_cache: Optional[GridLayout] = None
        self.selected_cell = -1
        self.hovered_cell = -1
        
        # Grid drag state
        self._drag_source_index = -1  # Index of cell being dragged
        self._drag_target_index = -1  # Current drop target index
        self._is_dragging = False     # True when dragging a grid tile
        self._drag_start_pos = None   # Mouse position at drag start
        
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        
        # UI background (not canvas)
        self.setStyleSheet("background-color: #1a1a1a;")
    
    def set_model(self, model: DataModel):
        """Set the data model and refresh."""
        self.model = model
        self.update_layout()
    
    def update_layout(self):
        """Recompute layout and repaint."""
        if self.model:
            # Layout is computed in virtual canvas space (export resolution)
            # The preview widget will scale it to fit during paintEvent
            canvas_w = self.model.grid_settings.canvas_width
            canvas_h = self.model.grid_settings.canvas_height
            self.layout_cache = self.model.compute_grid_layout(canvas_w, canvas_h)
        self.update()
    
    def set_selected_cell(self, cell_index: int):
        """Set the selected cell."""
        self.selected_cell = cell_index
        self.update()
    
    def _get_canvas_rect(self) -> QRectF:
        """Get the rectangle where the canvas is drawn (maintaining aspect ratio)."""
        if not self.model:
            return QRectF(0, 0, self.width(), self.height())
        
        gs = self.model.grid_settings
        canvas_aspect = gs.canvas_width / gs.canvas_height if gs.canvas_height > 0 else 16/9
        
        w = self.width()
        h = self.height()
        widget_aspect = w / h if h > 0 else 1
        
        if widget_aspect > canvas_aspect:
            # Widget is wider - fit to height
            canvas_h = h
            canvas_w = h * canvas_aspect
        else:
            # Widget is taller - fit to width
            canvas_w = w
            canvas_h = w / canvas_aspect
        
        # Center canvas in widget
        x = (w - canvas_w) / 2
        y = (h - canvas_h) / 2
        
        return QRectF(x, y, canvas_w, canvas_h)
    
    def _cell_at_pos(self, pos: QPointF) -> int:
        """Find which cell is at the given position. Returns -1 if none."""
        if not self.model or not self.layout_cache:
            return -1
        
        canvas_rect = self._get_canvas_rect()
        layout = self.layout_cache
        gs = self.model.grid_settings
        scale = canvas_rect.width() / gs.canvas_width if gs.canvas_width > 0 else 1
        
        # Convert widget pos to canvas pos
        canvas_x = (pos.x() - canvas_rect.x()) / scale
        canvas_y = (pos.y() - canvas_rect.y()) / scale
        
        for i, cell in enumerate(layout.cells):
            if not cell.is_visible:
                continue
            
            # Cell position with global offset
            cx = cell.x + layout.offset_x
            cy = cell.y + layout.offset_y
            
            if (cx <= canvas_x <= cx + cell.width and 
                cy <= canvas_y <= cy + cell.height):
                return i
        
        return -1
    
    def _get_reordered_clip_indices(self) -> List[int]:
        """
        Get clip indices as they would appear after the current drag operation.
        Used for visual reflow preview during drag.
        """
        if not self.model or self._drag_source_index < 0 or self._drag_target_index < 0:
            # No drag in progress - return normal order
            return list(range(len(self.model.clips))) if self.model else []
        
        n = len(self.model.clips)
        if self._drag_source_index >= n or self._drag_target_index >= n:
            return list(range(n))
        
        if self._drag_source_index == self._drag_target_index:
            return list(range(n))
        
        # Simulate the reorder
        indices = list(range(n))
        source = self._drag_source_index
        target = self._drag_target_index
        
        # Remove from source and insert at target
        moved = indices.pop(source)
        indices.insert(target, moved)
        
        return indices
    
    def paintEvent(self, event):
        """Paint the canvas and grid cells with video thumbnails."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        if not self.model:
            return
        
        gs = self.model.grid_settings
        layout = self.layout_cache
        
        if not layout:
            return
        
        # Get canvas area
        canvas_rect = self._get_canvas_rect()
        scale = canvas_rect.width() / gs.canvas_width if gs.canvas_width > 0 else 1
        
        # Draw canvas background
        bg_color = QColor(gs.background_color)
        painter.fillRect(canvas_rect, bg_color)
        
        # Get clip order (may be reordered during drag preview)
        if self._is_dragging and self._drag_target_index >= 0:
            reordered_indices = self._get_reordered_clip_indices()
        else:
            reordered_indices = None
        
        # Draw grid cells
        for i, cell in enumerate(layout.cells):
            if not cell.is_visible:
                continue
                
            # Cell position with global offset, scaled
            cx = canvas_rect.x() + (cell.x + layout.offset_x) * scale
            cy = canvas_rect.y() + (cell.y + layout.offset_y) * scale
            cw = cell.width * scale
            ch = cell.height * scale
            
            cell_rect = QRectF(cx, cy, cw, ch)
            
            # Determine which clip to show in this cell (may be reordered during drag)
            if reordered_indices and i < len(reordered_indices):
                clip_idx = reordered_indices[i]
            else:
                clip_idx = cell.clip_index
            
            # Get clip and thumbnail
            clip = None
            thumbnail = None
            if clip_idx >= 0 and clip_idx < len(self.model.clips):
                clip = self.model.clips[clip_idx]
                thumbnail = get_thumbnail_pixmap(clip.thumbnail_path)
            
            # Draw thumbnail or placeholder background
            if thumbnail:
                # Scale thumbnail to fit cell while maintaining aspect ratio
                scaled_pixmap = thumbnail.scaled(
                    int(cw), int(ch),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                # Center the thumbnail in the cell
                px = cx + (cw - scaled_pixmap.width()) / 2
                py = cy + (ch - scaled_pixmap.height()) / 2
                painter.drawPixmap(int(px), int(py), scaled_pixmap)
            else:
                # No thumbnail - draw placeholder
                painter.fillRect(cell_rect, QColor("#404040"))
            
            # Draw drag feedback overlays
            if self._is_dragging:
                # Highlight drop target
                if i == self._drag_target_index:
                    painter.fillRect(cell_rect, QColor(0, 200, 100, 80))  # Green highlight
                    painter.setPen(QPen(QColor("#00cc66"), 3))
                    painter.drawRect(cell_rect)
                # Dim the source cell
                elif i == self._drag_source_index and self._drag_target_index != self._drag_source_index:
                    painter.fillRect(cell_rect, QColor(0, 0, 0, 100))  # Dimmed overlay
            elif i == self.selected_cell:
                # Selected cell - blue overlay
                painter.fillRect(cell_rect, QColor(51, 153, 255, 100))  # Semi-transparent blue
                painter.setPen(QPen(QColor("#0066cc"), 3))
                painter.drawRect(cell_rect)
            elif i == self.hovered_cell:
                # Hovered cell - light overlay
                painter.fillRect(cell_rect, QColor(255, 255, 255, 40))  # Semi-transparent white
                painter.setPen(QPen(QColor("#808080"), 1))
                painter.drawRect(cell_rect)
            
            # Draw clip name at bottom of cell (overlay on thumbnail)
            if clip:
                name = clip.display_name
                if len(name) > 20:
                    name = name[:17] + "..."
                
                # Draw semi-transparent background for text
                text_bg_height = max(16, int(20 * scale))
                text_rect = QRectF(cx, cy + ch - text_bg_height, cw, text_bg_height)
                painter.fillRect(text_rect, QColor(0, 0, 0, 160))  # Semi-transparent black
                
                # Draw text
                painter.setPen(QColor("white"))
                font = painter.font()
                font.setPointSize(max(7, int(9 * scale)))
                painter.setFont(font)
                painter.drawText(text_rect, Qt.AlignCenter, name)
        
        painter.end()
    
    def resizeEvent(self, event):
        """Handle resize - recompute layout."""
        super().resizeEvent(event)
        self.update_layout()
    
    def mousePressEvent(self, event):
        """Handle mouse press for cell selection."""
        if event.button() == Qt.LeftButton:
            cell_index = self._cell_at_pos(event.position())
            if cell_index >= 0:
                self.selected_cell = cell_index
                self._drag_start_pos = event.position()
                self._drag_source_index = cell_index
                self.cell_clicked.emit(cell_index)
                self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for hover effect and drag detection."""
        # Check if we should start a drag operation
        if (event.buttons() & Qt.LeftButton and 
            self._drag_start_pos is not None and 
            self._drag_source_index >= 0 and
            self._drag_source_index < len(self.model.clips) if self.model else False):
            
            # Check drag distance threshold
            delta = event.position() - self._drag_start_pos
            if delta.manhattanLength() > 10:  # Start drag after 10px movement
                self._is_dragging = True
                self._drag_target_index = self._drag_source_index
        
        # Update drag target during drag
        if self._is_dragging:
            cell_index = self._cell_at_pos(event.position())
            # Only update if over a valid cell
            if cell_index >= 0 and cell_index < len(self.model.clips) if self.model else False:
                if cell_index != self._drag_target_index:
                    self._drag_target_index = cell_index
                    self.update()
        else:
            # Normal hover behavior
            cell_index = self._cell_at_pos(event.position())
            if cell_index != self.hovered_cell:
                self.hovered_cell = cell_index
                self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - commit drag or cancel."""
        if event.button() == Qt.LeftButton and self._is_dragging:
            # Commit the reorder if valid
            if (self._drag_source_index >= 0 and 
                self._drag_target_index >= 0 and 
                self._drag_source_index != self._drag_target_index and
                self.model and
                self._drag_source_index < len(self.model.clips) and
                self._drag_target_index < len(self.model.clips)):
                
                # Emit reorder signal
                self.clip_reordered.emit(self._drag_source_index, self._drag_target_index)
        
        # Reset drag state
        self._is_dragging = False
        self._drag_source_index = -1
        self._drag_target_index = -1
        self._drag_start_pos = None
        self.update()
    
    def leaveEvent(self, event):
        """Clear hover when mouse leaves."""
        self.hovered_cell = -1
        # Cancel drag if mouse leaves widget
        if self._is_dragging:
            self._is_dragging = False
            self._drag_source_index = -1
            self._drag_target_index = -1
            self._drag_start_pos = None
        self.update()
    
    def dragEnterEvent(self, event):
        """Accept drag if it contains clip data (from Import list)."""
        if event.mimeData().hasFormat("application/x-clip-index"):
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """Update hover during external drag (from Import list)."""
        if event.mimeData().hasFormat("application/x-clip-index"):
            cell_index = self._cell_at_pos(event.position())
            if cell_index != self.hovered_cell:
                self.hovered_cell = cell_index
                self.update()
            event.acceptProposedAction()
    
    def dragLeaveEvent(self, event):
        """Clear hover when drag leaves."""
        self.hovered_cell = -1
        self.update()
    
    def dropEvent(self, event):
        """Handle drop of clip from Import list onto cell (swap operation)."""
        if event.mimeData().hasFormat("application/x-clip-index"):
            data = event.mimeData().data("application/x-clip-index")
            stream = QDataStream(data, QIODevice.ReadOnly)
            clip_index = stream.readInt32()
            
            cell_index = self._cell_at_pos(event.position())
            if cell_index >= 0:
                self.clip_dropped.emit(clip_index, cell_index)
            
            event.acceptProposedAction()
        
        self.hovered_cell = -1
        self.update()


# =============================================================================
# IMPORT LIST WIDGET
# =============================================================================

class ImportListWidget(QListWidget):
    """Custom list widget for imported clips with drag-drop reordering."""
    
    clip_reordered = Signal(int, int)  # Emits (from_index, to_index)
    clip_remove_requested = Signal(int)  # Emits clip index to remove
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable drag-drop for internal reordering
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Track drag source for reorder detection
        self._drag_start_row = -1
        
        self.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #404040;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #3399ff;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
        """)
    
    def mimeTypes(self):
        """Return supported MIME types for drag-and-drop."""
        return ["application/x-qabstractitemmodeldatalist", "application/x-clip-index"]
    
    def mimeData(self, items):
        """Create MIME data for drag operation - supports both internal and external drops."""
        mime_data = super().mimeData(items)
        
        # Add custom MIME data for external drops (to grid)
        if items:
            row = self.row(items[0])
            data = QByteArray()
            stream = QDataStream(data, QIODevice.WriteOnly)
            stream.writeInt32(row)
            mime_data.setData("application/x-clip-index", data)
            self._drag_start_row = row
        
        return mime_data
    
    def dropEvent(self, event):
        """Handle drop for internal reordering."""
        if event.source() != self:
            # Not an internal move
            event.ignore()
            return
        
        # Get source and drop positions
        source_row = self._drag_start_row
        drop_row = self.indexAt(event.position().toPoint()).row()
        
        # If dropping at the end or on invalid position
        if drop_row == -1:
            drop_row = self.count()
        
        # Prevent default behavior - we'll handle the reorder via model
        event.setDropAction(Qt.IgnoreAction)
        event.accept()
        
        # Emit signal to update model - list will be rebuilt from model
        if source_row != -1 and source_row != drop_row:
            # Adjust drop position if dragging down
            if source_row < drop_row:
                drop_row -= 1
            
            self.clip_reordered.emit(source_row, drop_row)
        
        self._drag_start_row = -1
    
    def _show_context_menu(self, pos):
        """Show right-click context menu."""
        item = self.itemAt(pos)
        if item:
            menu = QMenu(self)
            remove_action = menu.addAction("Remove clip")
            action = menu.exec_(self.mapToGlobal(pos))
            if action == remove_action:
                row = self.row(item)
                self.clip_remove_requested.emit(row)


# =============================================================================
# IMPORT TAB
# =============================================================================

class ImportTab(QWidget):
    """Tab for managing imported video clips."""
    
    files_added = Signal(list)  # Emits list of file paths
    clip_selected = Signal(int)  # Emits clip index
    clip_reordered = Signal(int, int)  # Emits (from, to)
    clip_remove_requested = Signal(int)  # Emits clip index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Add files button
        self.add_btn = QPushButton("Add Files…")
        self.add_btn.clicked.connect(self._on_add_files)
        layout.addWidget(self.add_btn)
        
        # Clip count label
        self.count_label = QLabel("0 clips imported")
        self.count_label.setStyleSheet("color: #808080;")
        layout.addWidget(self.count_label)
        
        # Clip list
        self.clip_list = ImportListWidget()
        self.clip_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.clip_list.clip_reordered.connect(self.clip_reordered)
        self.clip_list.clip_remove_requested.connect(self.clip_remove_requested)
        layout.addWidget(self.clip_list)
    
    def _on_add_files(self):
        """Open file dialog to add video files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*.*)"
        )
        if files:
            self.files_added.emit(files)
    
    def _on_selection_changed(self):
        """Handle list selection change."""
        row = self.clip_list.currentRow()
        self.clip_selected.emit(row)
    
    def add_clip(self, display_name: str):
        """Add a clip to the list."""
        self.clip_list.addItem(display_name)
        self._update_count()
    
    def remove_clip(self, index: int):
        """Remove a clip from the list."""
        if 0 <= index < self.clip_list.count():
            self.clip_list.takeItem(index)
            self._update_count()
    
    def set_selected(self, index: int):
        """Set the selected clip in the list."""
        self.clip_list.blockSignals(True)
        if 0 <= index < self.clip_list.count():
            self.clip_list.setCurrentRow(index)
        else:
            self.clip_list.clearSelection()
        self.clip_list.blockSignals(False)
    
    def clear(self):
        """Clear all clips from the list."""
        self.clip_list.clear()
        self._update_count()
    
    def _update_count(self):
        """Update the clip count label."""
        count = self.clip_list.count()
        self.count_label.setText(f"{count} clip{'s' if count != 1 else ''} imported")


# =============================================================================
# VIDEO TAB
# =============================================================================

class VideoTab(QWidget):
    """Tab for video/grid settings with full layout controls."""
    
    settings_changed = Signal()  # Emitted when any setting changes
    swap_requested = Signal()    # Emitted when swap button clicked
    reset_auto_tile_requested = Signal()  # Emitted when reset auto tile button clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False  # Prevent recursive signals
        self._manual_tile_edit = False  # Track if user manually edited tile size
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Grid Size Group
        grid_group = QGroupBox("Grid Size")
        grid_layout = QFormLayout(grid_group)
        
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(2)
        self.rows_spin.valueChanged.connect(self._on_setting_changed)
        grid_layout.addRow("Rows:", self.rows_spin)
        
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(2)
        self.cols_spin.valueChanged.connect(self._on_setting_changed)
        grid_layout.addRow("Columns:", self.cols_spin)
        
        self.auto_grid_check = QCheckBox("Auto-adjust grid")
        self.auto_grid_check.setChecked(True)
        self.auto_grid_check.setToolTip("Automatically adjust rows/columns based on clip count")
        grid_layout.addRow(self.auto_grid_check)
        
        layout.addWidget(grid_group)
        
        # Tile Size Group
        tile_group = QGroupBox("Tile Size")
        tile_layout = QFormLayout(tile_group)
        
        # Auto Tile Size checkbox
        self.auto_tile_check = QCheckBox("Auto Tile Size")
        self.auto_tile_check.setChecked(True)
        self.auto_tile_check.setToolTip("Automatically compute tile size to fit export resolution")
        self.auto_tile_check.stateChanged.connect(self._on_auto_tile_changed)
        tile_layout.addRow(self.auto_tile_check)
        
        self.tile_width_spin = QSpinBox()
        self.tile_width_spin.setRange(64, 3840)
        self.tile_width_spin.setValue(320)
        self.tile_width_spin.setSingleStep(16)
        self.tile_width_spin.valueChanged.connect(self._on_tile_width_changed)
        tile_layout.addRow("Width:", self.tile_width_spin)
        
        self.tile_height_spin = QSpinBox()
        self.tile_height_spin.setRange(36, 2160)
        self.tile_height_spin.setValue(180)
        self.tile_height_spin.setSingleStep(9)
        self.tile_height_spin.valueChanged.connect(self._on_tile_height_changed)
        tile_layout.addRow("Height:", self.tile_height_spin)
        
        self.lock_aspect_check = QCheckBox("Lock 16:9 aspect")
        self.lock_aspect_check.setChecked(True)
        tile_layout.addRow(self.lock_aspect_check)
        
        # Reset Auto Tile Size button
        self.reset_tile_btn = QPushButton("Reset Auto Tile Size")
        self.reset_tile_btn.setToolTip("Recompute tile size to fit canvas")
        self.reset_tile_btn.clicked.connect(self._on_reset_auto_tile)
        tile_layout.addRow(self.reset_tile_btn)
        
        # Initially disable manual tile inputs when auto is enabled
        self._update_tile_inputs_state()
        
        layout.addWidget(tile_group)
        
        # Gap Settings Group
        gap_group = QGroupBox("Gaps")
        gap_layout = QFormLayout(gap_group)
        
        self.gap_h_spin = QSpinBox()
        self.gap_h_spin.setRange(0, 100)
        self.gap_h_spin.setValue(4)
        self.gap_h_spin.valueChanged.connect(self._on_setting_changed)
        gap_layout.addRow("Horizontal:", self.gap_h_spin)
        
        self.gap_v_spin = QSpinBox()
        self.gap_v_spin.setRange(0, 100)
        self.gap_v_spin.setValue(4)
        self.gap_v_spin.valueChanged.connect(self._on_setting_changed)
        gap_layout.addRow("Vertical:", self.gap_v_spin)
        
        layout.addWidget(gap_group)
        
        # Appearance Group
        appear_group = QGroupBox("Appearance")
        appear_layout = QFormLayout(appear_group)
        
        # Background color
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setFixedHeight(24)
        self._bg_color = "#000000"
        self._update_color_button()
        self.bg_color_btn.clicked.connect(self._on_pick_color)
        appear_layout.addRow("Background:", self.bg_color_btn)
        
        # Scaling mode
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItem("Letterbox (fit)", "letterbox")
        self.scaling_combo.addItem("Crop (fill)", "crop")
        self.scaling_combo.addItem("Stretch", "stretch")
        self.scaling_combo.currentIndexChanged.connect(self._on_setting_changed)
        appear_layout.addRow("Scaling:", self.scaling_combo)
        
        layout.addWidget(appear_group)
        
        # Layout Group
        layout_group = QGroupBox("Layout")
        layout_form = QFormLayout(layout_group)
        
        # Anchor position
        self.anchor_combo = QComboBox()
        self.anchor_combo.addItem("Center", "center")
        self.anchor_combo.addItem("Top-Left", "top_left")
        self.anchor_combo.addItem("Top-Right", "top_right")
        self.anchor_combo.addItem("Bottom-Left", "bottom_left")
        self.anchor_combo.addItem("Bottom-Right", "bottom_right")
        self.anchor_combo.currentIndexChanged.connect(self._on_setting_changed)
        layout_form.addRow("Anchor:", self.anchor_combo)
        
        # Center last row
        self.center_last_row_check = QCheckBox("Center last row")
        self.center_last_row_check.setChecked(True)
        self.center_last_row_check.stateChanged.connect(self._on_setting_changed)
        layout_form.addRow(self.center_last_row_check)
        
        layout.addWidget(layout_group)
        
        # Swap button
        self.swap_btn = QPushButton("Swap Selected Cells")
        self.swap_btn.setEnabled(False)
        self.swap_btn.clicked.connect(self.swap_requested)
        layout.addWidget(self.swap_btn)
        
        layout.addStretch()
    
    def _on_setting_changed(self):
        """Emit settings changed signal."""
        if not self._updating:
            self.settings_changed.emit()
    
    def _on_auto_tile_changed(self, state):
        """Handle Auto Tile Size checkbox change."""
        if self._updating:
            return
        
        is_auto = self.auto_tile_check.isChecked()
        self._update_tile_inputs_state()
        
        # If enabling auto, trigger recompute
        if is_auto:
            self._manual_tile_edit = False
            self.reset_auto_tile_requested.emit()
        
        self._on_setting_changed()
    
    def _update_tile_inputs_state(self):
        """Enable/disable tile inputs based on auto tile state."""
        is_auto = self.auto_tile_check.isChecked()
        # When auto is enabled, make inputs read-only (shows computed values)
        self.tile_width_spin.setEnabled(not is_auto)
        self.tile_height_spin.setEnabled(not is_auto)
    
    def _on_tile_width_changed(self, value):
        """Handle tile width change - maintain 16:9 if locked."""
        if self._updating:
            return
        
        # Only mark as manual edit if auto tile is enabled
        # (if auto is disabled, user is already in manual mode)
        if self.auto_tile_check.isChecked():
            self.auto_tile_check.setChecked(False)
            self._manual_tile_edit = True
        
        if self.lock_aspect_check.isChecked():
            self._updating = True
            self.tile_height_spin.setValue(int(value * 9 / 16))
            self._updating = False
        self._on_setting_changed()
    
    def _on_tile_height_changed(self, value):
        """Handle tile height change - maintain 16:9 if locked."""
        if self._updating:
            return
        
        # Only mark as manual edit if auto tile is enabled
        # (if auto is disabled, user is already in manual mode)
        if self.auto_tile_check.isChecked():
            self.auto_tile_check.setChecked(False)
            self._manual_tile_edit = True
        
        if self.lock_aspect_check.isChecked():
            self._updating = True
            self.tile_width_spin.setValue(int(value * 16 / 9))
            self._updating = False
        self._on_setting_changed()
    
    def _on_reset_auto_tile(self):
        """Reset to auto tile sizing."""
        self._updating = True
        self.auto_tile_check.setChecked(True)
        self._manual_tile_edit = False
        self._update_tile_inputs_state()
        self._updating = False
        self.reset_auto_tile_requested.emit()
    
    def is_manual_tile_edit(self) -> bool:
        """Check if user manually edited tile size."""
        return self._manual_tile_edit
    
    def clear_manual_tile_edit(self):
        """Clear manual tile edit flag (called after auto tile reset)."""
        self._manual_tile_edit = False
    
    def _on_pick_color(self):
        """Open color picker for background."""
        color = QColorDialog.getColor(QColor(self._bg_color), self, "Background Color")
        if color.isValid():
            self._bg_color = color.name()
            self._update_color_button()
            self._on_setting_changed()
    
    def _update_color_button(self):
        """Update the color button appearance."""
        self.bg_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._bg_color};
                border: 1px solid #606060;
            }}
        """)
    
    def get_settings(self) -> GridSettings:
        """Get current settings as GridSettings object."""
        return GridSettings(
            rows=self.rows_spin.value(),
            columns=self.cols_spin.value(),
            tile_width=self.tile_width_spin.value(),
            tile_height=self.tile_height_spin.value(),
            gap_horizontal=self.gap_h_spin.value(),
            gap_vertical=self.gap_v_spin.value(),
            background_color=self._bg_color,
            scaling_mode=self.scaling_combo.currentData(),
            anchor=self.anchor_combo.currentData(),
            center_last_row=self.center_last_row_check.isChecked(),
            auto_tile_size=self.auto_tile_check.isChecked()
        )
    
    def set_settings(self, settings: GridSettings):
        """Set UI from GridSettings object (programmatic update - does not disable auto)."""
        self._updating = True
        
        self.rows_spin.setValue(settings.rows)
        self.cols_spin.setValue(settings.columns)
        self.tile_width_spin.setValue(settings.tile_width)
        self.tile_height_spin.setValue(settings.tile_height)
        self.gap_h_spin.setValue(settings.gap_horizontal)
        self.gap_v_spin.setValue(settings.gap_vertical)
        
        self._bg_color = settings.background_color
        self._update_color_button()
        
        # Set scaling mode combo
        idx = self.scaling_combo.findData(settings.scaling_mode)
        if idx >= 0:
            self.scaling_combo.setCurrentIndex(idx)
        
        # Set anchor combo
        idx = self.anchor_combo.findData(settings.anchor)
        if idx >= 0:
            self.anchor_combo.setCurrentIndex(idx)
        
        self.center_last_row_check.setChecked(settings.center_last_row)
        
        # Set auto tile size checkbox
        self.auto_tile_check.setChecked(settings.auto_tile_size)
        self._update_tile_inputs_state()
        
        self._updating = False
    
    def is_auto_grid_enabled(self) -> bool:
        """Check if auto-grid adjustment is enabled."""
        return self.auto_grid_check.isChecked()


# =============================================================================
# EXPORT TAB
# =============================================================================

class ExportTab(QWidget):
    """Tab for export settings and controls."""
    
    settings_changed = Signal()  # Emitted when any setting changes
    export_requested = Signal()  # Emitted when export button clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False  # Prevent recursive signals
        self._clip_count = 0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Resolution Group
        res_group = QGroupBox("Resolution")
        res_layout = QFormLayout(res_group)
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("1080p (1920×1080)", "1080p")
        self.resolution_combo.addItem("1440p / 2K (2560×1440)", "1440p")
        self.resolution_combo.addItem("4K (3840×2160)", "4k")
        self.resolution_combo.addItem("Custom", "custom")
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        res_layout.addRow("Preset:", self.resolution_combo)
        
        # Custom resolution inputs
        self.custom_res_widget = QWidget()
        custom_res_layout = QHBoxLayout(self.custom_res_widget)
        custom_res_layout.setContentsMargins(0, 0, 0, 0)
        
        self.custom_width_spin = QSpinBox()
        self.custom_width_spin.setRange(320, 7680)
        self.custom_width_spin.setValue(1920)
        self.custom_width_spin.setSingleStep(16)
        self.custom_width_spin.valueChanged.connect(self._on_setting_changed)
        custom_res_layout.addWidget(self.custom_width_spin)
        
        custom_res_layout.addWidget(QLabel("×"))
        
        self.custom_height_spin = QSpinBox()
        self.custom_height_spin.setRange(180, 4320)
        self.custom_height_spin.setValue(1080)
        self.custom_height_spin.setSingleStep(9)
        self.custom_height_spin.valueChanged.connect(self._on_setting_changed)
        custom_res_layout.addWidget(self.custom_height_spin)
        
        res_layout.addRow("Size:", self.custom_res_widget)
        self.custom_res_widget.setVisible(False)
        
        layout.addWidget(res_group)
        
        # FPS Group
        fps_group = QGroupBox("Frame Rate")
        fps_layout = QFormLayout(fps_group)
        
        self.fps_auto_check = QCheckBox("Auto (from first clip)")
        self.fps_auto_check.setChecked(True)
        self.fps_auto_check.stateChanged.connect(self._on_fps_auto_changed)
        fps_layout.addRow(self.fps_auto_check)
        
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._on_setting_changed)
        self.fps_spin.setEnabled(False)
        fps_layout.addRow("Manual:", self.fps_spin)
        
        layout.addWidget(fps_group)
        
        # Quality Group
        quality_group = QGroupBox("Export Speed")
        quality_layout = QFormLayout(quality_group)
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Best Quality (slow, CRF 17)", "best")
        self.quality_combo.addItem("Balanced (medium, CRF 19)", "balanced")
        self.quality_combo.addItem("Fast Export (veryfast, CRF 25)", "fast")
        self.quality_combo.setCurrentIndex(1)  # Default to Balanced
        self.quality_combo.currentIndexChanged.connect(self._on_setting_changed)
        quality_layout.addRow("Preset:", self.quality_combo)
        
        layout.addWidget(quality_group)
        
        # Performance Group
        performance_group = QGroupBox("Performance")
        performance_layout = QVBoxLayout(performance_group)
        
        self.use_proxies_check = QCheckBox("Use proxy files for faster export")
        self.use_proxies_check.setChecked(True)
        self.use_proxies_check.stateChanged.connect(self._on_setting_changed)
        performance_layout.addWidget(self.use_proxies_check)
        
        proxy_info = QLabel(
            "Proxies are lower-resolution copies of your clips cached for faster processing.\n"
            "They are generated once and reused. Recommended for large grids or high-res videos."
        )
        proxy_info.setWordWrap(True)
        proxy_info.setStyleSheet("color: #888; font-size: 10px;")
        performance_layout.addWidget(proxy_info)
        
        layout.addWidget(performance_group)
        
        # Duration Group
        duration_group = QGroupBox("Duration")
        duration_layout = QFormLayout(duration_group)
        
        self.duration_combo = QComboBox()
        self.duration_combo.addItem("Trim to shortest clip", "shortest")
        self.duration_combo.addItem("Custom duration (loops if needed)", "custom")
        self.duration_combo.currentIndexChanged.connect(self._on_duration_mode_changed)
        duration_layout.addRow("Mode:", self.duration_combo)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 86400.0)  # Up to 24 hours
        self.duration_spin.setValue(10.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(" sec")
        self.duration_spin.valueChanged.connect(self._on_setting_changed)
        self.duration_spin.setEnabled(False)
        duration_layout.addRow("Duration:", self.duration_spin)
        
        layout.addWidget(duration_group)
        
        # Audio Group
        audio_group = QGroupBox("Audio")
        audio_layout = QFormLayout(audio_group)
        
        self.audio_combo = QComboBox()
        self.audio_combo.addItem("None", "none")
        self.audio_combo.addItem("First clip", "first")
        self.audio_combo.addItem("Selected clip", "selected")
        self.audio_combo.currentIndexChanged.connect(self._on_audio_source_changed)
        audio_layout.addRow("Source:", self.audio_combo)
        
        self.audio_clip_combo = QComboBox()
        self.audio_clip_combo.currentIndexChanged.connect(self._on_setting_changed)
        self.audio_clip_combo.setEnabled(False)
        audio_layout.addRow("Clip:", self.audio_clip_combo)
        
        layout.addWidget(audio_group)
        
        # Export Button
        self.export_btn = QPushButton("Export Video")
        self.export_btn.setMinimumHeight(36)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d7d2d;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d9d3d;
            }
            QPushButton:pressed {
                background-color: #1d5d1d;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #606060;
            }
        """)
        layout.addWidget(self.export_btn)
        
        layout.addStretch()
    
    def _on_setting_changed(self):
        """Emit settings changed signal."""
        if not self._updating:
            self.settings_changed.emit()
    
    def _on_resolution_changed(self, index):
        """Handle resolution preset change."""
        is_custom = self.resolution_combo.currentData() == "custom"
        self.custom_res_widget.setVisible(is_custom)
        self._on_setting_changed()
    
    def _on_fps_auto_changed(self, state):
        """Handle FPS auto checkbox change."""
        self.fps_spin.setEnabled(state != Qt.Checked)
        self._on_setting_changed()
    
    def _on_duration_mode_changed(self, index):
        """Handle duration mode change."""
        is_custom = self.duration_combo.currentData() == "custom"
        self.duration_spin.setEnabled(is_custom)
        self._on_setting_changed()
    
    def _on_audio_source_changed(self, index):
        """Handle audio source change."""
        is_selected = self.audio_combo.currentData() == "selected"
        self.audio_clip_combo.setEnabled(is_selected)
        self._on_setting_changed()
    
    def _on_export_clicked(self):
        """Handle export button click."""
        self.export_requested.emit()
    
    def update_clip_list(self, clips: List[Clip]):
        """Update the audio clip selection combo box."""
        self._updating = True
        current_idx = self.audio_clip_combo.currentIndex()
        self.audio_clip_combo.clear()
        
        for i, clip in enumerate(clips):
            self.audio_clip_combo.addItem(f"{i+1}. {clip.display_name}", i)
        
        self._clip_count = len(clips)
        
        # Restore selection if valid
        if 0 <= current_idx < len(clips):
            self.audio_clip_combo.setCurrentIndex(current_idx)
        elif len(clips) > 0:
            self.audio_clip_combo.setCurrentIndex(0)
        
        # Enable export only if there are clips
        self.export_btn.setEnabled(len(clips) > 0)
        
        self._updating = False
    
    def get_settings(self) -> ExportSettings:
        """Get current settings as ExportSettings object."""
        return ExportSettings(
            resolution_preset=self.resolution_combo.currentData(),
            custom_width=self.custom_width_spin.value(),
            custom_height=self.custom_height_spin.value(),
            fps_auto=self.fps_auto_check.isChecked(),
            fps_manual=self.fps_spin.value(),
            quality_preset=self.quality_combo.currentData(),
            use_proxies=self.use_proxies_check.isChecked(),
            duration_mode=self.duration_combo.currentData(),
            custom_duration=self.duration_spin.value(),
            audio_source=self.audio_combo.currentData(),
            audio_clip_index=self.audio_clip_combo.currentData() or 0
        )
    
    def set_settings(self, settings: ExportSettings):
        """Set UI from ExportSettings object."""
        self._updating = True
        
        # Resolution
        idx = self.resolution_combo.findData(settings.resolution_preset)
        if idx >= 0:
            self.resolution_combo.setCurrentIndex(idx)
        self.custom_width_spin.setValue(settings.custom_width)
        self.custom_height_spin.setValue(settings.custom_height)
        self.custom_res_widget.setVisible(settings.resolution_preset == "custom")
        
        # FPS
        self.fps_auto_check.setChecked(settings.fps_auto)
        self.fps_spin.setValue(settings.fps_manual)
        self.fps_spin.setEnabled(not settings.fps_auto)
        
        # Quality/Speed
        idx = self.quality_combo.findData(settings.quality_preset)
        if idx >= 0:
            self.quality_combo.setCurrentIndex(idx)
        
        # Performance
        self.use_proxies_check.setChecked(settings.use_proxies)
        
        # Duration
        idx = self.duration_combo.findData(settings.duration_mode)
        if idx >= 0:
            self.duration_combo.setCurrentIndex(idx)
        self.duration_spin.setValue(settings.custom_duration)
        self.duration_spin.setEnabled(settings.duration_mode == "custom")
        
        # Audio
        idx = self.audio_combo.findData(settings.audio_source)
        if idx >= 0:
            self.audio_combo.setCurrentIndex(idx)
        idx = self.audio_clip_combo.findData(settings.audio_clip_index)
        if idx >= 0:
            self.audio_clip_combo.setCurrentIndex(idx)
        self.audio_clip_combo.setEnabled(settings.audio_source == "selected")
        
        self._updating = False


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Ezgrid")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)
        
        # Initialize data model
        self.model = DataModel()
        self.current_file: Optional[str] = None
        
        # Sync grid canvas dimensions with export resolution on startup
        canvas_w, canvas_h = self.model.export_settings.get_resolution()
        self.model.grid_settings.canvas_width = canvas_w
        self.model.grid_settings.canvas_height = canvas_h
        
        # Compute initial tile sizes
        if self.model.grid_settings.auto_tile_size:
            self.model._auto_compute_tile_size()
        
        # Initialize undo stack
        self.undo_stack = UndoStack()
        
        # Track cells for swap operation
        self._first_selected_cell = -1
        
        self._setup_menu()
        self._setup_ui()
        self._connect_signals()
        self._sync_settings_to_ui()
        self._update_undo_redo_actions()
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: white;
            }
            QMenuBar::item:selected {
                background-color: #3a3a3a;
            }
            QMenu {
                background-color: #2d2d2d;
                color: white;
            }
            QMenu::item:selected {
                background-color: #3399ff;
            }
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: white;
                padding: 8px 16px;
                border: 1px solid #404040;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #505050;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #606060;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #404040;
                padding: 4px;
            }
            QLabel {
                color: white;
            }
            QGroupBox {
                color: white;
                border: 1px solid #404040;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QCheckBox {
                color: white;
            }
        """)
    
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save As…", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self._on_undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)
        
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(self._on_redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)
        
        # Import menu
        import_menu = menubar.addMenu("Import")
        
        add_files_action = QAction("Add Files…", self)
        add_files_action.setShortcut("Ctrl+I")
        add_files_action.triggered.connect(lambda: self.import_tab._on_add_files())
        import_menu.addAction(add_files_action)
    
    def _setup_ui(self):
        """Setup the main UI."""
        # Central widget with vertical layout (main content + details panel)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        
        # Main content area with horizontal splitter
        main_content = QWidget()
        main_layout = QHBoxLayout(main_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Canvas preview (center)
        self.canvas_preview = CanvasPreview()
        self.canvas_preview.set_model(self.model)
        splitter.addWidget(self.canvas_preview)
        
        # Right sidebar with tabs
        self.sidebar = QTabWidget()
        self.sidebar.setMinimumWidth(280)
        self.sidebar.setMaximumWidth(400)
        
        self.import_tab = ImportTab()
        self.sidebar.addTab(self.import_tab, "Import")
        
        self.video_tab = VideoTab()
        self.sidebar.addTab(self.video_tab, "Video")
        
        self.export_tab = ExportTab()
        self.sidebar.addTab(self.export_tab, "Export")
        
        splitter.addWidget(self.sidebar)
        
        # Set splitter sizes (70% preview, 30% sidebar)
        splitter.setSizes([700, 300])
        
        outer_layout.addWidget(main_content, stretch=1)
        
        # Details panel at the bottom (collapsible logs area)
        self.details_panel = DetailsPanel()
        outer_layout.addWidget(self.details_panel)
    
    def _connect_signals(self):
        """Connect all signals."""
        # Import tab signals
        self.import_tab.files_added.connect(self._on_files_added)
        self.import_tab.clip_selected.connect(self._on_clip_selected)
        self.import_tab.clip_reordered.connect(self._on_clip_reordered)
        self.import_tab.clip_remove_requested.connect(self._on_clip_remove)
        
        # Video tab signals
        self.video_tab.settings_changed.connect(self._on_settings_changed)
        self.video_tab.swap_requested.connect(self._on_swap_cells)
        self.video_tab.reset_auto_tile_requested.connect(self._on_reset_auto_tile)
        
        # Canvas preview signals
        self.canvas_preview.cell_clicked.connect(self._on_cell_selected)
        self.canvas_preview.clip_dropped.connect(self._on_clip_dropped_to_cell)
        self.canvas_preview.clip_reordered.connect(self._on_grid_clip_reordered)
        
        # Export tab signals
        self.export_tab.settings_changed.connect(self._on_export_settings_changed)
        self.export_tab.export_requested.connect(self._on_export_requested)
    
    def _update_undo_redo_actions(self):
        """Update the enabled state of undo/redo actions."""
        self.undo_action.setEnabled(self.undo_stack.can_undo())
        self.redo_action.setEnabled(self.undo_stack.can_redo())
    
    def _on_undo(self):
        """Handle undo action."""
        if self.undo_stack.undo():
            self._refresh_import_list()
            self._refresh_preview()
            self._update_undo_redo_actions()
    
    def _on_redo(self):
        """Handle redo action."""
        if self.undo_stack.redo():
            self._refresh_import_list()
            self._refresh_preview()
            self._update_undo_redo_actions()
    
    def _sync_settings_to_ui(self):
        """Sync model settings to UI."""
        self.video_tab.set_settings(self.model.grid_settings)
        self.export_tab.set_settings(self.model.export_settings)
        self.export_tab.update_clip_list(self.model.clips)
        self.canvas_preview.update_layout()
    
    def _sync_ui_to_settings(self):
        """Sync UI settings to model."""
        ui_settings = self.video_tab.get_settings()
        
        # If auto-grid is disabled, use UI values for rows/cols
        if not self.video_tab.is_auto_grid_enabled():
            self.model.grid_settings.rows = ui_settings.rows
            self.model.grid_settings.columns = ui_settings.columns
        
        # Sync auto_tile_size from UI checkbox
        self.model.grid_settings.auto_tile_size = ui_settings.auto_tile_size
        
        # Always copy other settings
        self.model.grid_settings.tile_width = ui_settings.tile_width
        self.model.grid_settings.tile_height = ui_settings.tile_height
        self.model.grid_settings.gap_horizontal = ui_settings.gap_horizontal
        self.model.grid_settings.gap_vertical = ui_settings.gap_vertical
        self.model.grid_settings.background_color = ui_settings.background_color
        self.model.grid_settings.scaling_mode = ui_settings.scaling_mode
        self.model.grid_settings.anchor = ui_settings.anchor
        self.model.grid_settings.center_last_row = ui_settings.center_last_row
    
    def _on_reset_auto_tile(self):
        """Handle reset auto tile size request."""
        self.model.grid_settings.auto_tile_size = True
        self.model._auto_compute_tile_size()
        self.video_tab.clear_manual_tile_edit()
        self._sync_settings_to_ui()
    
    def _refresh_preview(self):
        """Refresh the canvas preview."""
        self.canvas_preview.update_layout()
    
    def _refresh_import_list(self):
        """Refresh the import list from model."""
        self.import_tab.clear()
        for clip in self.model.clips:
            self.import_tab.add_clip(clip.display_name)
        self.export_tab.update_clip_list(self.model.clips)
    
    def _on_files_added(self, file_paths: list):
        """Handle adding new video files with background import."""
        if not file_paths:
            return
        
        # For a single file, do it inline (fast enough)
        if len(file_paths) == 1:
            clip = Clip.from_path(file_paths[0])
            self.model.add_clip(clip)
            self.import_tab.add_clip(clip.display_name)
            self.export_tab.update_clip_list(self.model.clips)
            self._sync_settings_to_ui()
            return
        
        # For multiple files, use background import with progress dialog
        dialog = ImportProgressDialog(file_paths, self)
        dialog.clips_ready.connect(self._on_import_complete)
        dialog.exec()
    
    def _on_import_complete(self, clips: List[Clip]):
        """Handle completion of background import."""
        for clip in clips:
            self.model.add_clip(clip)
            self.import_tab.add_clip(clip.display_name)
        
        # Update export tab clip list
        self.export_tab.update_clip_list(self.model.clips)
        
        # Sync auto-adjusted grid to UI
        self._sync_settings_to_ui()
    
    def _on_clip_selected(self, clip_index: int):
        """Handle clip selection in import list - sync to grid."""
        self.model.selected_clip_index = clip_index
        
        # In this model, clip index = cell index
        cell_index = self.model.get_cell_for_clip(clip_index)
        self.model.selected_cell_index = cell_index
        self.canvas_preview.set_selected_cell(cell_index)
        
        # Reset swap tracking
        self._first_selected_cell = -1
        self.video_tab.swap_btn.setEnabled(False)
    
    def _on_cell_selected(self, cell_index: int):
        """Handle cell selection in grid - sync to import list."""
        self.model.selected_cell_index = cell_index
        
        # Find corresponding clip (cell_index = clip_index in row-major order)
        clip_index = self.model.get_clip_at_cell(cell_index)
        self.model.selected_clip_index = clip_index
        self.import_tab.set_selected(clip_index)
        
        # Swap logic - track two selections
        if self._first_selected_cell == -1:
            self._first_selected_cell = cell_index
            self.video_tab.swap_btn.setEnabled(False)
        elif self._first_selected_cell != cell_index:
            # Second cell selected, enable swap
            self.video_tab.swap_btn.setEnabled(True)
        else:
            # Same cell clicked again
            self._first_selected_cell = -1
            self.video_tab.swap_btn.setEnabled(False)
    
    def _on_swap_cells(self):
        """Swap the two selected cells (actually swaps clips in import list)."""
        if self._first_selected_cell >= 0 and self.model.selected_cell_index >= 0:
            # Swap clips at these positions
            idx_a = self._first_selected_cell
            idx_b = self.model.selected_cell_index
            if idx_a < len(self.model.clips) and idx_b < len(self.model.clips):
                # Use undo stack for swap
                cmd = SwapClipsCommand(self.model, idx_a, idx_b)
                self.undo_stack.push(cmd)
                self._refresh_import_list()
                self._refresh_preview()
                self._update_undo_redo_actions()
            
            self._first_selected_cell = -1
            self.video_tab.swap_btn.setEnabled(False)
    
    def _on_clip_reordered(self, from_index: int, to_index: int):
        """Handle clip reorder in import list - update model with undo support."""
        # Use undo stack for reorder
        cmd = ReorderClipsCommand(self.model, from_index, to_index)
        self.undo_stack.push(cmd)
        
        # Rebuild import list from model to ensure synchronization
        self._refresh_import_list()
        
        # Refresh grid preview
        self._refresh_preview()
        self._update_undo_redo_actions()
        
        # Clear selection to avoid confusion
        self.model.selected_clip_index = -1
        self.model.selected_cell_index = -1
        self.canvas_preview.set_selected_cell(-1)
    
    def _on_grid_clip_reordered(self, from_index: int, to_index: int):
        """Handle clip reorder from grid drag - update model with undo support."""
        # Use undo stack for reorder
        cmd = ReorderClipsCommand(self.model, from_index, to_index)
        self.undo_stack.push(cmd)
        
        # Rebuild import list from model to ensure synchronization
        self._refresh_import_list()
        
        # Refresh grid preview
        self._refresh_preview()
        self._update_undo_redo_actions()
        
        # Keep selection following the dragged item
        self.model.selected_clip_index = to_index
        self.model.selected_cell_index = to_index
        self.canvas_preview.set_selected_cell(to_index)
        self.import_tab.set_selected(to_index)
    
    def _on_clip_remove(self, clip_index: int):
        """Handle clip removal request with undo support."""
        if 0 <= clip_index < len(self.model.clips):
            clip = self.model.clips[clip_index]
            cmd = RemoveClipCommand(self.model, clip_index, clip)
            self.undo_stack.push(cmd)
            
            # Rebuild import list and update
            self._refresh_import_list()
            self.export_tab.update_clip_list(self.model.clips)
            self._sync_settings_to_ui()
            self._update_undo_redo_actions()
    
    def _on_settings_changed(self):
        """Handle settings change from Video tab."""
        old_tile_size = (self.model.grid_settings.tile_width, self.model.grid_settings.tile_height)
        old_gaps = (self.model.grid_settings.gap_horizontal, self.model.grid_settings.gap_vertical)
        old_grid = (self.model.grid_settings.rows, self.model.grid_settings.columns)
        
        self._sync_ui_to_settings()
        
        # If auto tile is enabled and grid/gaps changed, recompute tiles
        if self.model.grid_settings.auto_tile_size:
            new_gaps = (self.model.grid_settings.gap_horizontal, self.model.grid_settings.gap_vertical)
            new_grid = (self.model.grid_settings.rows, self.model.grid_settings.columns)
            
            if old_gaps != new_gaps or old_grid != new_grid:
                self.model._auto_compute_tile_size()
                self.video_tab.set_settings(self.model.grid_settings)
        
        self._refresh_preview()
    
    def _on_export_settings_changed(self):
        """Handle export settings change from Export tab."""
        old_resolution = self.model.export_settings.get_resolution()
        self.model.export_settings = self.export_tab.get_settings()
        new_resolution = self.model.export_settings.get_resolution()
        
        print(f"[EXPORT SETTINGS] Resolution changed: {old_resolution} -> {new_resolution}")
        
        # If resolution changed and auto tile size is enabled, recompute tile sizes
        if old_resolution != new_resolution and self.model.grid_settings.auto_tile_size:
            print(f"[EXPORT SETTINGS] Auto-recomputing tile sizes...")
            self.model._auto_compute_tile_size()
            # Update grid tab UI with new tile sizes
            self.video_tab.set_settings(self.model.grid_settings)
            self._refresh_preview()
    
    def _on_export_requested(self):
        """
        Handle export button click - check FFmpeg availability and run export.
        
        Flow:
        1. Check if clips exist
        2. Check FFmpeg availability (show dialog if not found)
        3. Generate proxies if requested
        4. Ask for output file location
        5. Generate FFmpeg command
        6. Run export with progress dialog
        """
        if not self.model.clips:
            QMessageBox.warning(self, "Export", "No clips to export.")
            return
        
        # Check if ffmpeg is available (bundled or system PATH)
        ffmpeg_path = find_ffmpeg()
        if not ffmpeg_path:
            # Show FFmpeg not found dialog with Download/Retry options
            dialog = FFmpegNotFoundDialog(self)
            result = dialog.exec()
            
            if result != QDialog.Accepted or not dialog.was_ffmpeg_found():
                return  # User cancelled or FFmpeg still not found
            
            # Re-check after successful retry
            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                return
        
        # Handle proxy generation if requested
        proxy_paths: Optional[List[str]] = None
        if self.model.export_settings.use_proxies:
            # Show proxy generation dialog (worker extracts dimensions from model)
            proxy_dialog = ProxyGenerationDialog(self.model, self)
            result = proxy_dialog.exec()
            
            if result == QDialog.Accepted:
                proxy_paths = proxy_dialog.get_proxy_paths()
                # Show summary message
                QMessageBox.information(
                    self, "Proxy Generation",
                    proxy_dialog.get_message()
                )
            else:
                # User cancelled proxy generation
                reply = QMessageBox.question(
                    self, "Continue Export?",
                    "Proxy generation was cancelled. Continue with export using original files?\n\n"
                    "This may be slower for large grids.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                # Continue without proxies
                proxy_paths = None
        
        # Ask for output file location
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "",
            "MP4 Video (*.mp4);;All Files (*.*)"
        )
        if not output_path:
            return
        
        if not output_path.lower().endswith('.mp4'):
            output_path += '.mp4'
        
        # Ensure tile sizes are up-to-date with current export resolution
        if self.model.grid_settings.auto_tile_size:
            self.model._auto_compute_tile_size()
        
        # Debug logging before FFmpeg export
        gs = self.model.grid_settings
        es = self.model.export_settings
        out_w, out_h = es.get_resolution()
        
        # Compute grid dimensions
        grid_w = gs.columns * gs.tile_width + max(0, gs.columns - 1) * gs.gap_horizontal
        grid_h = gs.rows * gs.tile_height + max(0, gs.rows - 1) * gs.gap_vertical
        
        # Compute centering offsets
        dx = (out_w - grid_w) / 2
        dy = (out_h - grid_h) / 2
        
        print(f"\n{'='*60}")
        print(f"[FFMPEG EXPORT] Debug Info:")
        print(f"  Resolution: {out_w}x{out_h}")
        print(f"  Grid: {gs.rows}x{gs.columns} (rows x cols)")
        print(f"  Gaps: {gs.gap_horizontal}x{gs.gap_vertical} (H x V)")
        print(f"  Tile Size: {gs.tile_width}x{gs.tile_height}")
        print(f"  Grid Size: {grid_w}x{grid_h}")
        print(f"  Centering Offset: dx={dx:.1f}, dy={dy:.1f}")
        print(f"  Auto Tile Size: {gs.auto_tile_size}")
        print(f"  Scaling Mode: {gs.scaling_mode}")
        print(f"{'='*60}\n")
        
        # Generate the FFmpeg command (with proxies if available)
        ffmpeg_cmd, filter_script_path = build_ffmpeg_command(
            self.model, output_path, proxy_paths
        )
        
        # Compute output duration for progress tracking
        output_duration = compute_output_duration(self.model.clips, self.model.export_settings)
        
        # If we couldn't determine duration, use a fallback
        if output_duration <= 0:
            output_duration = 60.0  # Default to 60 seconds for progress estimation
        
        # Run export with progress dialog
        progress_dialog = ExportProgressDialog(
            ffmpeg_cmd, output_duration, output_path, filter_script_path, self
        )
        progress_dialog.exec()
        
        # Show result
        if progress_dialog.was_successful():
            QMessageBox.information(
                self, "Export Complete",
                f"Video exported successfully!\n\n{output_path}"
            )
        else:
            message = progress_dialog.get_message()
            if "cancelled" in message.lower():
                # Don't show error for user cancellation
                pass
            else:
                QMessageBox.warning(
                    self, "Export Failed",
                    f"Export failed:\n\n{message}"
                )
    
    def _on_clip_dropped_to_cell(self, from_clip_index: int, to_cell_index: int):
        """Handle dropping a clip onto a grid cell - swaps positions with undo support."""
        if from_clip_index >= 0 and to_cell_index >= 0:
            if from_clip_index < len(self.model.clips) and to_cell_index < len(self.model.clips):
                cmd = SwapClipsCommand(self.model, from_clip_index, to_cell_index)
                self.undo_stack.push(cmd)
                self._refresh_import_list()
                self._refresh_preview()
                self._update_undo_redo_actions()
    
    def _on_new(self):
        """Create a new project."""
        reply = QMessageBox.question(
            self, "New Project",
            "Create a new project? Unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.model.clear()
            self.undo_stack.clear()
            self.current_file = None
            self._refresh_import_list()
            self._sync_settings_to_ui()
            self._update_undo_redo_actions()
            self.setWindowTitle("Ezgrid")
    
    def _on_open(self):
        """Open an existing project."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "Ezgrid Project (*.ezgrid);;All Files (*.*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.model = DataModel.from_dict(data)
                self.canvas_preview.set_model(self.model)
                self.undo_stack.clear()
                self.current_file = file_path
                self._refresh_import_list()
                self._sync_settings_to_ui()
                self._update_undo_redo_actions()
                self.setWindowTitle(f"Ezgrid - {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file:\n{e}")
    
    def _on_save(self):
        """Save the current project."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self._on_save_as()
    
    def _on_save_as(self):
        """Save the project to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            "Ezgrid Project (*.ezgrid);;All Files (*.*)"
        )
        if file_path:
            if not file_path.endswith('.ezgrid'):
                file_path += '.ezgrid'
            self._save_to_file(file_path)
    
    def _save_to_file(self, file_path: str):
        """Save model to file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.model.to_dict(), f, indent=2)
            self.current_file = file_path
            self.setWindowTitle(f"Ezgrid - {Path(file_path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(42, 42, 42))
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(58, 58, 58))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(51, 153, 255))
    palette.setColor(QPalette.Highlight, QColor(51, 153, 255))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
