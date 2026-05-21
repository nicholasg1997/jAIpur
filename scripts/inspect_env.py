from pprint import pprint

from jaipur_rl.training.env_spec import build_env_spec


def main() -> None:
    spec = build_env_spec()
    print("Jaipur RL environment summary:")
    pprint(spec)


if __name__ == "__main__":
    main()