# demos.py

import argparse
import json
from experiments.config import load_and_apply_config # config.py의 load_config 함수 임포트

def _main():
    # 1. ArgumentParser 생성
    parser = argparse.ArgumentParser(
        description="Run robot environment demos with dynamic configurations.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- 필수 인자 ---
    parser.add_argument("--env_name", type=str, required=True, help="Name of the environment config file to load (e.g., 'cleanup_table').")
    parser.add_argument("--arm_type", type=str, default="right", choices=['right', 'left', 'dual'], help="Type of the robot arm setup.")

    # --- 덮어쓸 파라미터 ---
    # 새로운 파라미터는 여기에 추가!
    # 규칙: --<섹션>--<키> 형태로 인자 추가
    # dest='env.max_episode_length' 처럼 .을 사용해 딕셔너리 경로 지정
    parser.add_argument("--env--max_episode_length", type=int, dest='env.max_episode_length', help="Override max_episode_length in env config.")
    parser.add_argument("--env--action_scale", type=float, nargs='+', dest='env.action_scale', help="Override action_scale (e.g., --env--action_scale 0.01 0.01 1.0).")
    parser.add_argument("--env--robot_prefix", type=str, dest='env.robot_prefix', help="Override robot_prefix in env config.")
    parser.add_argument("--train--discount", type=float, dest='train.discount', help="Override discount factor in train config.")

    args = parser.parse_args()

    # 2. 커맨드 라인에서 입력된 값만 모아서 덮어쓰기용 딕셔너리 생성
    cli_overrides = {key: value for key, value in vars(args).items()
                     if value is not None and '.' in key}

    print("="*50)
    print(f"Starting demo for env='{args.env_name}', arm='{args.arm_type}'")
    if cli_overrides:
        print("Received command-line overrides:", cli_overrides)
    print("="*50 + "\n")


    # 3. load_config에 overrides 딕셔너리 전달
    env, final_config = load_config(
        env_name=args.env_name,
        arm_type=args.arm_type,
        overrides=cli_overrides
    )

    print("\n" + "="*50)
    print("✅ Final Config after all overrides:")
    # 보기 쉽게 JSON 형태로 최종 설정 출력
    print(json.dumps(final_config, indent=2, ensure_ascii=False))
    print("="*50)

    # ... 이후 환경을 사용하는 로직 ...
    print("\n🎉 Demo setup complete. You can now use the 'env' object.")
    # env.reset()
    # for _ in range(10):
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load environment from YAML configs.")
    parser.add_argument("--env_name", type=str, required=True, help="e.g., 'cleanup_table'")
    parser.add_argument("--arm_type", type=str, default="right", choices=['right', 'left', 'dual'])
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
    print(train_config_instance)
    
if __name__ == "__main__":
    main()