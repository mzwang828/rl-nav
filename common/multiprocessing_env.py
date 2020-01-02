import multiprocessing
multiprocessing.set_start_method('spawn', True)
from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "get_goal":
            for grid in env.grid.grid:
                if grid is not None and grid.type == "goal":
                    goal_pose = grid.cur_pos
            conn.send(goal_pose)
        elif cmd == "agent_pose":
            agent_pose = env.agent_pos
            conn.send(agent_pose)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results
    
    def get_goal(self):
        for grid in self.envs[0].grid.grid:
            if grid is not None and grid.type == "goal":
                goal_pose = grid.cur_pos
        for local in self.locals:
            local.send(("get_goal", None))
        results = [goal_pose] + [local.recv() for local in self.locals]
        return results

    def agent_pose(self):
        for local in self.locals:
            local.send(("agent_pose", None))
        results = [self.envs[0].agent_pos] + [local.recv() for local in self.locals]
        return results

    def render(self):
        raise NotImplementedError