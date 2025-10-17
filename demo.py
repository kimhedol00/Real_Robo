# demos.py

import argparse
import json
from experiments.config import load_and_apply_config # config.py의 load_config 함수 임포트
import os

from tqdm import tqdm
import datetime, copy
import numpy as np
import pickle as pkl
from pynput import keyboard

skip_key = False
save_key = False
def on_press(key):
    global skip_key
    global save_key
    try:
        if key.char == 's':
            skip_key = True
        elif key.char == 'p':
            save_key = True
    except AttributeError:
        pass


listener = keyboard.Listener(
    on_press=on_press)
listener.start()



def save_traj(transitions, success_needed, _name):

    _date = datetime.datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(f"./demos/{_name}"):
        os.makedirs(f"./demos/{_name}")
        
    if not os.path.exists(f"./demos/{_name}/{_date}"):
        os.makedirs(f"./demos/{_name}/{_date}")

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demos/{_name}/{_date}/{_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")




def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load environment from YAML configs.")
    parser.add_argument("--env_name", type=str, required=True, help="e.g., 'cleanup_table'")
    parser.add_argument("--arm_type", type=str, default="right", choices=['right', 'left', 'dual'])
    parser.add_argument("--success", type=int, default=10)
    parser.add_argument("--fake_env", action="store_true", help="Run without physical robot.")
    
    # ✨ action_scale과 hz를 위한 인자 추가 ✨
    parser.add_argument("--action_scale", type=float, nargs='+', help="Override action scale. e.g., --action_scale 0.05 0.05 2.0")
    parser.add_argument("--hz", type=int, help="Override environment frequency (Hz).")
    
    args = parser.parse_args()

    # CLI 인자를 overrides 딕셔너리로 변환
    overrides = {}
    if args.action_scale:
        overrides['action_scale'] = args.action_scale
    if args.hz:
        overrides['hz'] = args.hz

    # 설정 로드 및 적용 후, TrainConfig 인스턴스 생성
    train_config_instance = load_and_apply_config(
        env_name=args.env_name,
        arm_type=args.arm_type,
        overrides=overrides
    )
    
    # get_environment 메서드 호출
    env = train_config_instance.get_environment(fake_env=args.fake_env)


    print(f"\n✅ Environment for '{args.env_name}' with '{args.arm_type}' arm(s) created successfully!")
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = args.success
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs


        global save_key 
        global skip_key

        if save_key:
            save_key=False
            print("save and quit")
            save_traj(transitions, success_count, args.env_name)
            exit()



        if done:
            # skip_key = False

            if hasattr(env, "stopwatch"):
                import pickle
                with open("timings.pkl", "wb") as f:
                    pickle.dump(env.stopwatch.elapsed, f)
            # if info["succeed"]:

            if skip_key:
                skip_key=False
                print('skip')
                trajectory = []
                returns = 0
                obs, info = env.reset()
                continue
            if len(trajectory)<=10:
                print("[Error] episode length!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! : ", len(trajectory))
                trajectory = []
                returns = 0
                obs, info = env.reset()
                continue
            for transition in trajectory:
                transitions.append(copy.deepcopy(transition))
            success_count += 1
            print("[Save]: episode length : ", len(trajectory))
            pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            

    save_traj(transitions, success_needed, args.env_name)

    

if __name__ == "__main__":
    main()
