import json
from pathlib import Path
from src.config import RAW_DIR
from typing import Any
import pandas as pd

def ensure_dir(path:Path):
    path.mkdir(parents=True,exist_ok=True)

def write_json(path:Path, data: Any)-> None:
    ensure_dir(path)
    try:
        with open(path,"w",encoding= "utf-8") as file:
            json.dump(data,file,indent =2)
    except TypeError as e:
        raise TypeError(f"Failed to serialise data to JSON at {path}: {e}")

#Returns the JSON at the path
def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        with open(path,"r",encoding = "utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {path}:{e}")
    
def write_df_csv(df: pd.DataFrame,path:Path):
    ensure_dir(path.parent)
    try:
        df.to_csv(path,index=False)
    except OSError as e:
        raise ValueError(f"Error writing DataFrame as CSV: {e}")

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error writing CSV at {path} to dataFrame: {e}")
    