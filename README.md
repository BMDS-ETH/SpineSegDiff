[user]: research-ai-templates
[repo]: SpineSegDiff 

[issues-shield]: https://img.shields.io/github/issues/BMDS-ETH/SpineSegDiffnnUnet
[issues-url]: https://github.com/BMDS-ETH/SpineSegDiffnnUnet/issues

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Issues][issues-shield]][issues-url]

<div align="center">

<h1 align="center"> Diffusion Models for Lumbar Spine Segmentation: SpineSegDiff </h1>

  <p align="center">
    [TEMPLATE DESCRIPTION]
    <br />
    <a href="https://github.com/BMDS-ETH/SpineSegDiffnnUnet"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.comBMDS-ETH/SpineSegDiffnnUnet">View Demo</a>
    ·
    <a href="https://github.com/BMDS-ETH/SpineSegDiffnnUnet/issues">Report Bug</a>
    ·
    <a href="https://github.com/BMDS-ETH/SpineSegDiffnnUnet/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



## 📋 Project Overview

Welcome to an automated Segmentation-Tool for multi modal Magnetic Resonance Images. The goal is to generate multiclass segmentations of T1-weighted and T2-weighted scans. The SPIDER Dataset is used for training the models.

### Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make dirs` or `make clean`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data.dvc           <- Keeps the raw data versioned.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── dvc.lock
    ├── dvc.yaml
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── docs               <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── results             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                 <- Generated graphics and figures to be used in reporting
    │   └── metrics.json             <- Relevant metrics after evaluating the model.
    │   └── training_metrics.json    <- Relevant metrics from training the model.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── __init__.py
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and architectures
    │   │   │   ├── metrics
    │   │   │   ├── modules
    │   │   │   └── networks
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── __init__.py
    │       └── utils.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

--------
<p align="right">[<a href="#readme-top">back to top</a>]</p>

<!-- GETTING STARTED -->
## Getting Started

Welcome to an automated Segmentation-Tool for multi modal Magnetic Resonance Images. The goal is to generate multiclass segmentations of T1-weighted and T2-weighted scans. The SPIDER Dataset is used for training the models.


### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* python >=3.8
* 
<p align="right">[<a href="#readme-top">back to top</a>]</p>

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import spinesegdiff

```

How to train the model:
_To train the model, please run the following command, you can change the parameters within the train.py file._

    python -u src\trainer.py  -e 150
    
    *** Default training parameteres ***
    parser.add_argument("-lr", help="set the learning rate for the unet", type=float, default=0.0001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=300)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=4)
    parser.add_argument("-pt", "--pretrained", help="whether to train from scratch or resume", action="store_true",
                        default=False)

<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature 3.1

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the Apache License Version 2.0, License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Maria Monzon - [Emai Me]( maria.monzonronda@hest.ethz.ch)

[Github Link]( https://github.com/BMDS-ETH/SpineSegDif)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

