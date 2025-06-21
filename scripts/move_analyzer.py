from jaipur_rl.envs.jaipur_env import JaipurEnv
from sb3_plus import MultiOutputPPO
from pathlib import Path
from pprint import pprint

env = JaipurEnv()
model_base_path = Path(__file__).resolve().parents[1] / "models"
model = MultiOutputPPO.load(model_base_path / "Jaipur_25.0M_scheduler_2025-06-20_18-23-16", env=None)
trained_agent = model


obs, _ = env.reset()
for j in range(200):
    action_batch, _ = trained_agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action_batch)
    pprint(action_batch)
    env.render()

    if done or truncated:
        p1_score = info.get('P1_Final_Score')
        print(f"p1 score: {p1_score}")
        p2_score = info.get('P2_Final_Score')
        print(f"p2 score: {p2_score}")
        if info['winner'] == 'P1':
            print("Player 1 Wins!")
        elif info['winner'] == 'P2':
            print("Player 2 Wins!")
        break
