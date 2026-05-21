import argparse
from pprint import pprint

from jaipur_rl.training.evaluate import (
    evaluate_model,
    evaluate_random_baseline,
    save_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.model:
        results = evaluate_model(args.model, args.episodes)
        print("\nModel Evaluation:")
    else:
        results = evaluate_random_baseline(args.episodes)
        print("\nRandom Baseline:")

    pprint(results)

    if args.output:
        save_results(results, args.output)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()