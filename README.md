# DDPG
A minimalist implementation of DDPG.

A single file [main.py](main.py) contains the entire implementation. Most control task environements in openAI gym should be compatible. However only InvertedPendulum-v2 and Reacher-v2 are tested.

Pe-trained networks for these two environments in the [saved_networks](saved_networks) folder.

### Setup

Setup intructions are written assuming that the mujoco simulator is already installed on your computer and that you have access to mujoco-gym environments.

1. Setup a virtualenv with python >= 3.9.x

```
virtualenv ddpg_env
source ddpg_env/bin/activate
```

2. Clone this repository

```
git clone https://github.com/Niwhskal/ddpg.git
cd ddpg/
```

3. Install requirements

```
pip3 install -r ./requirements.txt
```

### Training

If you wish to train an agent in an specific gym environment, use the following code template:

```
python3 main.py --env *your_gym_env_name* --seed *seed_value* --save_dir *specify_directory_to_save_weights* 
```

As an example: To train ddpg in the InvertedPendulum-v2 environment:

```
python3 main.py --env InvertedPendulum-v2 --seed 2324 --save_dir ./results/
```
This is sufficient to train without hassle. However, if you wish to tinker, you can provide command line arguments for a number of other parameters. For all of these please check [main.py](main.py)


### Inference

Once trained, you will have weights saved in [saved_networks/](saved_networks) folder. If not you need to place them in *saved_networks/\*your_env_name\*/* folder.

you can run inference using:

```
python3 main.py --env *your_env_name* --run_mode "eval" --seed *seed_value* --save_dir *path_to_weights*
```

as an example: to run inference using the pre-trained weights given in this repository:

```
python3 main.py --env InvertedPendulum-v2 --run_mode "eval" --seed 32423 --save_dir ./saved_networks/InvertedPendulum-v2/
```

This will render InvertedPendulum-v2 in a new window


### Misc

Tiny changes to the original ddpg algorithm were necessary for ideal results. A description of all changes and training details are provided in a report in this repository [ddpg_lakshwin.pdf](ddpg_lakshwin.pdf)

The file [main.py](main.py) is also thoroughly documented to ensure straigtforward understanding.