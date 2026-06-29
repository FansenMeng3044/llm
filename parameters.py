class EnvParams:
    SPECIES_AGENTS_RANGE = (3, 3)
    SPECIES_RANGE = (3, 5)
    TASKS_RANGE = (15, 50)
    MAX_TIME = 200
    TRAIT_DIM = 5
    DECISION_DIM = 30


class TrainParams:
    USE_GPU = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
    NUM_META_AGENT = 16
    LR = 1e-5
    GAMMA = 1
    DECAY_STEP = 2e3
    RESET_OPT = False
    EVALUATE = True
    MAX_EPISODE = 40000     # stop training after this many dispatched episodes (set high for "unbounded")
    EVALUATION_SAMPLES = 256
    RESET_RAY = False
    INCREASE_DIFFICULTY = 20000
    SUMMARY_WINDOW = 8
    DEMON_RATE = 0.5
    IL_DECAY = -1e-5  # -1e-6 700k decay 0.5, -1e-5 70k decay 0.5, -1e-4 7k decay 0.5
    BATCH_SIZE = 2048
    AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
    TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM
    EMBEDDING_DIM = 128
    SAMPLE_SIZE = 200
    PADDING_SIZE = 50
    POMO_SIZE = 10
    FORCE_MAX_OPEN_TASK = False


class SaverParams:
    FOLDER_NAME = 'save_1'
    MODEL_PATH = f'model/{FOLDER_NAME}'
    TRAIN_PATH = f'train/{FOLDER_NAME}'
    GIFS_PATH = f'gifs/{FOLDER_NAME}'
    LOAD_MODEL = False
    LOAD_FROM = 'current'  # 'best'
    SAVE = True
    SAVE_IMG = True
    SAVE_IMG_GAP = 1000

