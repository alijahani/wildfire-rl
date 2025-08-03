# Wildfire RL

**Wildfire RL** is a reinforcement learning project for wildfire suppression agents, built using PyTorch.

This project was developed for the **MOASEI 2025 Competition** (part of the AAMAS 2025 Competitions), and was awarded as the **winner of Track #3 (Wildfire)**.

## Team Members

* Hossein Savari
* Ali Jahani
* Afsaneh Habibi

## Getting Started

### Train the Agent

Note: Training was done by using a much larger parallel-envs variable.

You can download pre-trained models from [this](https://alijahani.home.kg/models/wildfire_rl.zip) link.

```bash
python main.py
```

### Test the Agent

```bash
python WildfireEvaluation.py --model-to-load 180 --testing_episodes 500 --seed 1 evalout "path/to/WS1.pkl"
```

## TODO

- [ ] Add a proper license

- [ ] Write a complete installation guide

- [ ] Add argparse with helpful CLI descriptions

- [ ] Refactor conv_agent module

- [ ] Clean up training logic

- [ ] Add proper storage for experience replay (and other configurations)

- [ ] Add experimentations for individual predictors

## License

Copyright (C) 2025 Ali Jahani

This project is currently unlicensed. I might add a copyleft license later.

Until then, all rights are reserved.
