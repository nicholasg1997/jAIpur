import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from jaipur_rl.envs.jaipur_env import JaipurEnv
from sb3_plus import MultiOutputPPO, make_multioutput_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from jaipur_rl.training.schedulers import linear_schedule, EntCoefSchedulerCallback
from datetime import datetime
from jaipur_rl.training.callbacks import InfoCallback

def train(steps: int = 1_000_000, n_envs: int = 16,
          init_lr: float = 1e-4, min_lr: float = 1e-6,
          init_ent_coef: float = 0.05, final_ent_coef: float = 0.005, ent_coef_frac: float = 0.4,
          pi_network=None, vf_network=None,
          tensorboard_log: str = "./jaipur_tensorboard/",
          log_name: str = "Jaipur_tb", save_path: str = "Models",
          save_name: str = None) -> None:
    if vf_network is None:
        vf_network = [512, 256]
    if pi_network is None:
        pi_network = [512, 256]
    print("running")
    print(f"Training with {steps:,} steps, LR {init_lr} → {min_lr}, EntCoef {init_ent_coef} → {final_ent_coef}")
    vec_env = make_multioutput_env(
        JaipurEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )

    lr_schedule = linear_schedule(init_lr, min_lr)

    print("Creating model...")
    model = MultiOutputPPO(
        policy='MIMOPolicy',
        env=vec_env,
        ent_coef=init_ent_coef,
        verbose=1,
        learning_rate=lr_schedule,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=dict(pi=pi_network, vf=vf_network),
        )
    )

    ent_coef_callback = EntCoefSchedulerCallback(
        initial_ent_coef=init_ent_coef,
        final_ent_coef=final_ent_coef,
        total_timesteps=int(steps * 0.95),
        decay_start_fraction=ent_coef_frac
    )

    info_logging_callback = InfoCallback()
    callback_list = [ent_coef_callback, info_logging_callback]

    model.learn(
        total_timesteps=steps,
        callback=callback_list,
        progress_bar=True,
        tb_log_name=log_name,
    )

    print("Saving model...")
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save_name is None:
        save_name = f"Jaipur_{steps/1_000_000}M_scheduler_{time}.zip"
    model.save(f"{save_path}/{save_name}")


if __name__ == "__main__":

    train(
        steps=25_000_000,
        n_envs=8,
        init_lr=1e-4,
        min_lr=1e-6,
        init_ent_coef=0.05,
        final_ent_coef=0.005,
        ent_coef_frac=0.4,
        pi_network=[256, 128],
        vf_network=[256, 256],
        tensorboard_log="./jaipur_tensorboard/",
        log_name="Jaipur_tb",
        save_path="models",
    )
