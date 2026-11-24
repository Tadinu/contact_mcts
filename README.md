# Contact Trajectory Optimization via MCTS in MuJoCo
This repo hosts the source code for the paper [Efficient Object Manipulation Planning with Monte Carlo Tree Search](https://arxiv.org/abs/2206.09023)

## Dependencies
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for rigid body dynamics (already in `requirements.txt`)
- [MuJoCo](https://github.com/google-deepmind/mujoco) for physical simulation (already in `requirements.txt`)
- `uv pip install -e robot_properties_nyu_finger` for NYU finger URDFs and configuration files
- `requirements.txt`

## Notebooks
We prepared a few notebooks
- ``/demos``
``mcts[miqp]_playground.ipynb`` shows step-by-step how to generate a contact plan for a given motion with force plots and plan visualization.
``mcts[miqp]_<robot_type>_simulation.ipynb`` simulates the contact plan for multi-articulators with impedance controllers in MuJoCo.
- ``/experiments`` contains notebooks to reproduce the respective experiments in the paper.
- ``/train`` contains notebooks to generate the training data and to train the neural networks.

## Citing

```
@article{zhu2022efficient,
  title={Efficient Object Manipulation Planning with Monte Carlo Tree Search},
  author={Zhu, Huaijiang and Meduri, Avadesh and Righetti, Ludovic},
  journal={arXiv preprint arXiv:2206.09023},
  year={2022}
}
```

## Maintainer
- Huaijiang Zhu
- Tadinu

## Copyrights

Copyright(c) 2023 New York University

## License

BSD 3-Clause License



