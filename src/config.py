from pathlib import Path

# ==============================================================================
# PROJECT & DIRECTORY CONFIGURATION
# ==============================================================================
# Absolute paths resolved dynamically relative to this file
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATASET_ROOT = PROJECT_ROOT / "Data set"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VIDEO_FRAMES_DIR = PROCESSED_DATA_DIR / "video_frames"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Manifest and inventory paths
VIDEO_INVENTORY_CSV = REPORTS_DIR / "video_inventory.csv"
FRAME_MANIFEST_CSV = PROCESSED_DATA_DIR / "frame_manifest.csv"
DATASET_CATALOG_CSV = PROCESSED_DATA_DIR / "dataset_catalog.csv"
VIDEO_EXTRACTION_REPORT_TXT = REPORTS_DIR / "video_extraction_report.txt"
PROCESSED_CLASS_DISTRIBUTION_CSV = REPORTS_DIR / "processed_class_distribution.csv"

# Ensure output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# VIDEO FRAME EXTRACTION CONFIGURATION
# ==============================================================================
# Options:
#   - "uniform_target": Extract TARGET_FRAMES_PER_VIDEO evenly spaced across duration
#   - "sample_fps": Extract frames at SAMPLE_FPS rate
FRAME_SAMPLING_MODE = "uniform_target"

# Number of representative frames to extract per video in uniform_target mode
TARGET_FRAMES_PER_VIDEO = 8

# Sampling rate when using "sample_fps" mode
SAMPLE_FPS = 2.0

# Bounds on frames per video
MIN_FRAMES_PER_VIDEO = 3
MAX_FRAMES_PER_VIDEO = 12

# Skip beginning and end temporal margin (percentage) to avoid camera shake / WhatsApp transitions
TEMPORAL_MARGIN_PERCENT = 0.05

# Image format and quality for saved extracted frames
FRAME_IMAGE_EXTENSION = ".jpeg"
JPEG_QUALITY = 95

# Reproducibility seed
RANDOM_SEED = 42

# ==============================================================================
# CNN INPUT & PREPROCESSING CONFIGURATION
# ==============================================================================
# Selected CNN input dimensions
# 224x224 is the gold standard for CNN architectures (ResNet, MobileNet, EfficientNet, Custom CNNs)
# balancing spatial feature preservation of dance mudras with computational tractability.
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
CNN_INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Options:
#   - "aspect_pad": Preserves aspect ratio by scaling max dimension to target and letterboxing (zero-padding).
#                   Crucial for Kuchipudi hand mudras to prevent geometric distortion of finger angles.
#   - "direct": Standard direct bilinear resizing to (IMAGE_HEIGHT, IMAGE_WIDTH).
RESIZE_METHOD = "aspect_pad"

# Options:
#   - "rescale_0_1": Scales pixel values from [0, 255] to [0.0, 1.0] (x / 255.0).
#   - "rescale_minus1_1": Scales pixel values to [-1.0, 1.0] ((x / 127.5) - 1.0).
#   - "standardize": Zero-center per-image: (x - mean) / std.
NORMALIZATION_METHOD = "rescale_0_1"

# Batch size for dataset generators
BATCH_SIZE = 32

# Results and visualization directories
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PREPROCESSING_EXAMPLES_DIR = PLOTS_DIR / "preprocessing_examples"
PREPROCESSING_REPORT_TXT = REPORTS_DIR / "preprocessing_report.txt"

# Ensure visualization directories exist
PREPROCESSING_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA AUGMENTATION CONFIGURATION (TRAINING DATA ONLY)
# ==============================================================================
# Master toggle for training augmentation
ENABLE_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.75  # 75% chance an image undergoes transformation

# Rotation: strictly bounded to preserve semantic mudra/gesture orientation
# Max +/- 8.0 degrees prevents changing gesture meaning (e.g. upright vs downward)
ROTATION_RANGE_DEGREES = 8.0

# Translation: subtle shifts (+/- 6%) to simulate camera framing variations
# Prevents clipping off fingertips, eye edges, or toes
WIDTH_SHIFT_FRACTION = 0.06
HEIGHT_SHIFT_FRACTION = 0.06

# Zoom: slight scale variation (+/- 5%) to simulate distance variation
ZOOM_RANGE = (0.95, 1.05)

# Photometric variations: subtle lighting differences simulating stage/studio light
BRIGHTNESS_RANGE = (0.90, 1.10)
CONTRAST_RANGE = (0.90, 1.10)

# Critical Kuchipudi Dance Rules:
# Left/right flipping is STRICTLY DISABLED to preserve lateral mudra and gaze semantics
HORIZONTAL_FLIP = False
VERTICAL_FLIP = False

# Minority class dynamic oversampling multiplier during training
MINORITY_CLASS_THRESHOLD = 5
MINORITY_OVERSAMPLE_FACTOR = 2

# Augmentation output artifacts
AUGMENTATION_EXAMPLES_DIR = PLOTS_DIR / "augmentation_examples"
AUGMENTATION_REPORT_TXT = REPORTS_DIR / "augmentation_report.txt"
AUGMENTATION_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CNN ARCHITECTURE & TRAINING CONFIGURATION
# ==============================================================================
MODEL_NAME = "KuchipudiNet_CustomCNN"
NUM_CLASSES = 75

# Hyperparameters
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.4
DENSE_UNITS = 256
L2_REGULARIZATION = 1e-4

# Optimization & Loss
LOSS_FUNCTION = "categorical_crossentropy"  # Supports "categorical_crossentropy" or "sparse_categorical_crossentropy"
OPTIMIZER_NAME = "adam"

# Training Callback Settings
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# Model checkpoint and persistence paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "kuchipudi_cnn_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "kuchipudi_cnn_final.keras"
CNN_ARCHITECTURE_REPORT_TXT = REPORTS_DIR / "cnn_architecture_report.txt"

# ==============================================================================
# TRANSFER LEARNING CONFIGURATION (STEP 8B)
# ==============================================================================
TRANSFER_MODEL_NAME = "Kuchipudi_MobileNetV3Large_Transfer"
PRETRAINED_BACKBONE = "MobileNetV3Large"  # Options: "MobileNetV3Large", "MobileNetV3Small", "EfficientNetB0"

# Two-Stage Learning Rates
LEARNING_RATE_STAGE1 = 1e-3  # Head-only training (backbone frozen)
LEARNING_RATE_STAGE2 = 1e-4  # Controlled end-to-end fine-tuning

# Stage Epochs
MAX_EPOCHS_STAGE1 = 20
MAX_EPOCHS_STAGE2 = 30

# Fine-tuning: Percentage of top backbone layers to unfreeze in Stage 2 (20-30%)
FINE_TUNE_PERCENTAGE = 0.25

# Checkpoint and Report paths for Transfer Learning
TRANSFER_BEST_MODEL_PATH = CHECKPOINT_DIR / "kuchipudi_transfer_best.keras"
TRANSFER_REPORT_TXT = REPORTS_DIR / "transfer_model_architecture_report.txt"

# ==============================================================================
# HIERARCHICAL CNN CONFIGURATION (STEP 10)
# ==============================================================================
HIERARCHICAL_MODEL_NAME = "Kuchipudi_Hierarchical_CNN"
NUM_CATEGORIES = 4
CATEGORY_NAMES = [
    "1. Hasthas",
    "2. Shiro Bhedhas (HEAD MOVEMENTS)",
    "3. Drishti Bhedhas(EYE MOVEMENTS)",
    "4. Pada Karma(Foot movements)"
]
CATEGORY_DISPLAY_NAMES = [
    "Hasthas",
    "Shiro Bhedhas",
    "Drishti Bhedhas",
    "Pada Karma"
]
CATEGORY_CLASS_COUNTS = {
    "1. Hasthas": 52,
    "2. Shiro Bhedhas (HEAD MOVEMENTS)": 9,
    "3. Drishti Bhedhas(EYE MOVEMENTS)": 8,
    "4. Pada Karma(Foot movements)": 6
}
HIERARCHICAL_REPORT_TXT = REPORTS_DIR / "hierarchical_architecture_report.txt"
HIERARCHICAL_MAPPING_JSON = PROCESSED_DATA_DIR / "hierarchical_class_mapping.json"

