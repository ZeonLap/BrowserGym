"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

import gymnasium as gym
import argparse
from pathlib import Path

from browsergym.experiments import ExpArgs, EnvArgs

from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs


def get_miniwob_tasks():
    import browsergym.miniwob  # register miniwob tasks as gym environments
    env_ids = [id for id in gym.envs.registry.keys() if id.startswith("browsergym/miniwob")]
    task_names = [id.split("/")[-1] for id in env_ids]
    return task_names


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    task_names = get_miniwob_tasks()

    for idx, task_name in enumerate(task_names):
        if idx >= 3:
            break
        env_args = EnvArgs(
            task_name=task_name,
            task_seed=None,
            max_steps=10,
            headless=True,
            slow_mo=500,
        )

        exp_args = ExpArgs(
            env_args=env_args,
            agent_args=GenericAgentArgs(
                chat_model_args=ChatModelArgs(
                    model_name="openai/gpt-4o",
                    max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                    max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
                    max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
                ),
                flags=Flags(
                    use_html=True,
                    use_ax_tree=True,
                    use_thinking=True,  # "Enable the agent with a memory (scratchpad)."
                    use_error_logs=True,  # "Prompt the agent with the error logs."
                    use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                    use_history=True,
                    use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                    use_past_error_logs=False,  # "Prompt the agent with the past error logs."
                    use_action_history=True,  # "Prompt the agent with the action history."
                    multi_actions=False,
                    action_space="bid",
                    use_abstract_example=True,  # "Prompt the agent with an abstract example."
                    use_concrete_example=True,  # "Prompt the agent with a concrete example."
                    use_screenshot=True,
                    enable_chat=False,
                    demo_mode="off",
                ),
            ),
        )

        exp_args.prepare(Path("./results"))
        exp_args.run()


if __name__ == "__main__":
    main()
