# execute items in command line 
import os 
from threading import Thread


def run_script(script):
    # env = "conda activate training; "
    cmd = script
    os.system(cmd)

# run training script

for rhos in [(0.01, 0.05, 0.1), (0.2, 0.4, 0.8)]:
    rho1, rho2, rho3 = rhos
    # start a new thread for each training run in env1 enviorment conda
    t = Thread(target=run_script, args=(f"python training_aug_master.py --rho {rho1} --device cuda:0 --run_name SAM_rho{rho1}",))
    t.start()
    t2 = Thread(target=run_script, args=(f"python training_aug_master.py --rho {rho2} --device cuda:1 --run_name SAM_rho{rho2}",))
    t2.start()
    t3 = Thread(target=run_script, args=(f"python training_aug_master.py --rho {rho3} --device cuda:2 --run_name SAM_rho{rho3}",))
    t3.start()
    
    # wait for all threads to finish
    t.join()
    t2.join()
    t3.join()

    # wait until training is done to start next set of training
    print("________________________Round complete_____________________________")
    