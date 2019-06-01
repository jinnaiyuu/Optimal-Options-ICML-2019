# Options for Planning and Reinforcement Learning.

Code for experiments on generating options for planning and reinforcement learning in our 2019 ICML papers:

[Jinnai Y. Park JW, Abel D, Konidaris G. 2019. Discovering Options for Exploration by Minimizing Cover Time. Proc. 36th International Conference on Machine Learning.](https://jinnaiyuu.github.io/pdf/papers/ICML-19-rl.pdf)

[Jinnai Y, Abel D, Hershkowitz E, Littman M, Konidaris G. 2019. Finding Options that Minimize Planning Time. Proc. 36th International Conference on Machine Learning](https://jinnaiyuu.github.io/pdf/papers/ICML-19-plan.pdf)


# Dependencies

The code is written in Python 3.
The code is dependent on numpy, scipy, and networkx.
To solve MOMI optimally, ortools is required.
[simple_rl](https://github.com/david-abel/simple_rl) is a library for running RL experiments developed by David Abel. As I made a few tweaks to it, I'm putting the whole code in this repository here.


# Directory

graph: approximation algorithms in graph algorithm literature.
option_generation: option generation algorithms.
experiments: Scripts to replicate experiments in papers.


# Example
```
python3 options/experiments/planning_experiments.py
python3 options/experiments/rl_experiments.py
```

# Author

Yuu Jinnai <yuu_jinnai@brown.edu>

Simple RL is developed by [David Abel](https://david-abel.github.io/).