import logging
import os
from datetime import datetime
LOG_FIlE=f"{datetime.now().strftime('%m_%d_%Y_%H_')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FIlE)
os.makedirs(logs_path,exist_ok=True)    
log_file_path=os.path.join(logs_path,LOG_FIlE)
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)

