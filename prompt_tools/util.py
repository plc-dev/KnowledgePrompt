from google.colab import drive
import os
import json
from typing import Any

class PersistenceLayer:
    def __init__(self):
        drive_path = "/content/drive"
        drive.mount(drive_path)
        self.base_directory = f"{drive_path}/My Drive/PromptRuns"

    def persist_dict(self, data: Dict[str, Any]) -> None:
        file_path = self.dir
        with open(file_path, "w", encoding="utf8") as f:
            json.dump(data, f)

    # /content/gdrive/My Drive/Output_folder
    def __create_folder(self, folder_path: str) -> None:
        try:
            os.mkdir(folder_path)
        except:
            print("Folder already exists")
