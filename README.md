# Energy Based Joint Embedding Predictive Architectures (JEPA)

EB JEPA is an open source library and tutorial aimed at learning representations using joint embedding predictive arhictectures and planning using them.
Examples include learning image (a), video (b), and action conidtioned video (c) predictive models representations, as well as planning with them (d).
![EB JEPA](docs/teaser.png)

## Examples

### Image JEPA

This example demonstrates learning self-supervised representations from unlabeled images on CIFAR 10, and evaluated on image classification.
![Moving MNIST](examples/image_jepa/assets/arch_figure.png)


### Video JEPA on [Moving MNIST](https://www.cs.toronto.edu/~nitish/unsupervised_video)

![Moving MNIST](examples/video_jepa/assets/viz.png)

In this toy setting, a world model is trained given the MNIST image representation (or a sequence)

Prediction targets:

- next state representation

### Action Conditioned Video JEPA on Two Rooms

This example demonstrates a Joint Embedding Predictive Architecture (JEPA) for action-conditioned world modeling in the Two Rooms environment. The model learns to predict future states based on current observations and actions. These representations enable planning towards a goal observation embedding.

| Planning Episode | Task Definition |
|------------------|-----------------|
| <img src="examples/ac_video_jepa/assets/top_randw_agent_steps_succ.gif" alt="Successful planning episode" width="155" /> | <img src="examples/ac_video_jepa/assets/top_randw_state.png" alt="Episode task definition" width="300" /> |
| *Successful planning episode* | *Episode task definition: from init to goal state* |

The world model is trained given:

- image representation (or a sequence)
- action

Prediction targets:

- next state representation

## Installation

We use [uv](https://docs.astral.sh/uv/guides/projects/) package manager to install and maintain packages. Once you have [installed uv](https://docs.astral.sh/uv/getting-started/installation/), run the following to create a new virtual environment.

```bash
uv sync
```

This will create a virtual environment within the project folder at `.venv/`. To activate this environment, run `source .venv/bin/activate`.

Alternatively, if you don't want to run activate everytime, you can just prepend `uv run` before your python scripts:

```bash
uv run python main.py
```

## Running test cases

Libraries added to eb-jepa [must have their own test cases](/tests/). To run the tests: `uv run pytest tests/`

## Development

- The uv package comes with `black` and `isort`, which must be run before adding any file in this repo. The continous integration will check the linting of the PRs and new files.
- Every PR should be reviewed by folks tagged at [CODEOWNERS](docs/CODEOWNERS).


## License
EB JEPA is Apache licensed, as found in the [LICENSE](LICENSE.md) file.
