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

from app import create_daily_report as c

if __name__ == "__main__":
    c.do_create(os.path.dirname(os.path.dirname(os.getcwd())))
