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
    MAX_EPISODE = 50000     # resume from 20k and train to here
    EVALUATION_SAMPLES = 256
    # --- online LLM reweighting loop ---
    ONLINE_REWEIGHT = True
    REWEIGHT_EVERY = 5000                                   # evaluate best model + reweight every N episodes
    REWEIGHT_TEST_ROOT = '/data/data2/mfs/llm/RALtest_5dist'  # pre-generated M2-M5 x 5-dist test sets
    DEEPSEEK_MODEL = 'deepseek-chat'                        # set to your exact DeepSeek model id (e.g. the "v4pro" one)
    REWEIGHT_FLOOR = 0.5                                    # min weight per distribution
    REWEIGHT_MAX_CHANGE = 3.0                               # max per-step weight change factor
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
    FOLDER_NAME = 'save_llm'          # new run dir so it does NOT overwrite the source checkpoint
    MODEL_PATH = f'model/{FOLDER_NAME}'
    TRAIN_PATH = f'train/{FOLDER_NAME}'
    GIFS_PATH = f'gifs/{FOLDER_NAME}'
    LOAD_MODEL = True                # resume from 20k
    LOAD_FROM = 'current'            # continue from current@20k (optimizer-consistent); baseline<-best file below
    # resume weights (absolute paths you already staged)
    LOAD_CHECKPOINT_PATH = '/data/data2/mfs/llm/model/save_1 copy/checkpoint_20000.pth'
    LOAD_BEST_PATH = '/data/data2/mfs/llm/model/save_1 copy/best_model_save_1_ep20000.pth'
    SAVE = True
    SAVE_IMG = True
    SAVE_IMG_GAP = 1000

