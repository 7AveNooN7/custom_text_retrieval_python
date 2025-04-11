from dataclasses import dataclass
from enum import Enum


class ConversionMethodEnum(Enum):
    SIMPLE = 'Simple'
    GROBID = 'Grobid'

    @staticmethod
    def is_enum_value(value: str) -> bool:
        try:
            ConversionMethodEnum(value)
            return True
        except ValueError:
            return False


@dataclass
class FileSettingsModel:
    conversion_method: ConversionMethodEnum
    use_filter: bool