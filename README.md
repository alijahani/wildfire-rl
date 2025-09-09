# Wildfire RL

**Wildfire RL** is a reinforcement learning project for wildfire suppression agents, built using PyTorch and free-range-zoo.

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

- [x] Add a proper license

- [ ] Write a complete installation guide

- [ ] Add argparse with helpful CLI descriptions

- [ ] Refactor conv_agent module

- [ ] Clean up training logic

- [ ] Add proper storage for experience replay (and other configurations)

- [ ] Add experimentations for individual predictors

## License

Copyright (C) 2025 Ali Jahani, Hossein Savari, Afsaneh Habibi

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see <https://www.gnu.org/licenses>.
