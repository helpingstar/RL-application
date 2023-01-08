"""
source : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
start_e  : starting epsilon value
end_e    : ending epsilon value
duration : number of steps to take from start_e to end_e
t        : current step number

ex)
start_e = 1.0
end_e = 0.1
exploration_ratio = 0.5
total_steps = 1000

linear_schedule(start_e, end_e, exploration_ratio * total_steps, t)
-> It decreases linearly from `start_e` to `end_e` by `exploration_ratio`.
    After that, it retains the value of `end_e`.

"""
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
