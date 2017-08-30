# Attend, Infer, Repeat: Fast Scene Understanding with Generative Models

This is an **unofficial** Tensorflow implementation of Attend, Infear, Repeat (AIR), as presented in the following paper:
[S. M. Ali Eslami](http://arkitus.com/) et. al., [Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](https://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models).

* **Author**: Adam Kosiorek, Oxford Robotics Institue, University of Oxford
* **Email**: adamk(at)robots.ox.ac.uk
* **Paper**: https://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models.pdf
* **Webpage**: http://ori.ox.ac.uk/

## Installation
Install [Tensorflow v1.1.0rc1](https://www.tensorflow.org/versions/r1.1/install/), [Sonnet v1.1](https://github.com/deepmind/sonnet/tree/3fd7d9d342d9683df83a44ebd048ef0d5668266b) and the following dependencies
 (using `pip install -r requirements.txt` (preferred) or `pip install [package]`):
* matplotlib==1.5.3
* numpy==1.12.1
* attrdict==2.0.0
* scipy==0.18.1

## Sample Results

AIR learns to reconstruct objects by painting them one by one in a blank canvas. The below figure comes from a model trained for 175k iterations; the maximum number of steps is set to 3, but there are never more than 2 objects.
The first row shows the input images, rows 2-4 are reconstructions at steps 1, 2 and 3 (with marked location of the attention glimpse in red, if it exists). Rows 4-7 are the reconstructed image crops, and above each crop is the
probability of executing 1, 2 or 3 steps. If the reconstructed crop is black and there is "0 with ..." written above it, it means that this step was not used.

![AIR results](https://raw.githubusercontent.com/akosiorek/attend_infer_repeat/master/results/progress_fig_175000.jpg "AIR results")

## Data  
Run `./scripts/create_dataset.sh`
The script creates train and validation datasets of multi-digit MNIST.

## Training
Run `./scripts/train_multi_mnist.sh`
The training script will run for 300k iteratios and will save model checkpoints and training progress figures every 10k iterations in `results/multi_mnist`. Tensorflow summaries are also stored in the same folder and Tensorboard can be used for monitoring.

The model seems to be very sensitive to initialisation. It might be necessary to run training multiple times before achieving count step accuracy close to the one reported in the paper.

## Experimentation
The jupyter notebook available at `attend_infer_repeat/experiment.ipynb` can be used for experimentation.

## Citation

If you find this repo useful in your research, please consider citing the original paper:

    @incollection{Eslami2016,
        title = {Attend, Infer, Repeat: Fast Scene Understanding with Generative Models},
        author = {Eslami, S. M. Ali and Heess, Nicolas and Weber, Theophane and Tassa, Yuval and Szepesvari, David and kavukcuoglu, koray and Hinton, Geoffrey E},
        booktitle = {Advances in Neural Information Processing Systems 29},
        editor = {D. D. Lee and M. Sugiyama and U. V. Luxburg and I. Guyon and R. Garnett},
        pages = {3225--3233},
        year = {2016},
        publisher = {Curran Associates, Inc.},
        url = {http://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models.pdf}
    }


## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see  <http://www.gnu.org/licenses/>.


## Release Notes
**Version 1.0**
* Original **unofficial** implementation; contains the multi-digit MNIST experiment.
