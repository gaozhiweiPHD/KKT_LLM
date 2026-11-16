"""Optional TRL environment_factory adapter, exposing one strictly checked tool."""
from .active_set_env import ActiveSetEnv
from .observation import observation
from .rewards import reward

class QPToolEnvironment:
    def __init__(self,corpus,reward_version="B",max_steps=20):
        self._corpus=corpus; self._reward_version=reward_version; self._max_steps=max_steps
        self._env=None

    def reset(self,problem_id,initial_active=None,**kwargs):
        # Dataset row repeated by TRL gives every member the identical task.
        self._env=ActiveSetEnv(self._corpus[problem_id],initial_active or [],self._max_steps)

    def apply_action(self,action: str,index: int=-1) -> str:
        """Apply exactly one active-set repair action and return numerical feedback.

        Args:
            action: One of ADD, DROP, or STOP.
            index: Constraint index for ADD/DROP; -1 for STOP.

        Returns:
            Current state and independently verified KKT residuals.
        """
        e=self._env
        if e.done: return "Episode terminated. No further actions accepted."
        if type(index) is not int or action not in ("ADD","DROP","STOP") or (action=="STOP" and index!=-1):
            text="INVALID_TOOL_ARGUMENTS"
        else: text="STOP" if action=="STOP" else f"{action}({index})"
        e.step(text)
        return observation(e)

    def get_reward(self):
        return reward(self._env,self._reward_version)
