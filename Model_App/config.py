import torch

# {'ResNet': 224}
MODEL_NAME = 'ResNet'
RESIZE_TO = 224
MODEL_PATH = './' + MODEL_NAME + '.pt'
BATCH_SIZE = 16
NUM_EPOCHS = 10
PRETRAINED_MODEL = './output/best_model.pt'
TRAIN_FOLDER_PATH = './data/train'
VALID_FOLDER_PATH = './data/valid'
TEST_FOLDER_PATH = './data/test'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASS = 10
CLASSES = [
    '101',
    '102',
    '208',
    '301',
    '310',
    'blue_dragon',
    'dormitory',
    'football_field',
    'library',
    'pepero'
]