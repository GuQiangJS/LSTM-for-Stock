import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
nb_dir = os.getcwd()
while(True):
    if nb_dir not in sys.path:
        sys.path.append(nb_dir)
    if nb_dir == os.path.dirname(nb_dir):
        break
    nb_dir = os.path.dirname(nb_dir)

from app import train_all as ta

if __name__ == "__main__":
    ta.do_train()
