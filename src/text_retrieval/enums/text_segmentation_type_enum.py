from enum import Enum


class TextSegmentationTypeEnum(Enum):
    CHARACTERS = 'Characters'
    TIK_TOKEN = 'Tokens (TikToken)'
    CURRENT_MODEL_TOKENIZER = 'Tokens (Current Model Tokenizer)'