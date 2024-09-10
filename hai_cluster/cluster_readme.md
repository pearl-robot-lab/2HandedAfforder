# Cluster usage
This document just summarizes tips & tricks to work with the hessian AI cluster "42". Thanks to Simon Kiefhaber (You can contact him via the Hessian AI slack channel). Before trying to connect to the cluster, make sure you have applied for access through your advisor and received an activation password to login to https://login01.ai.tu-darmstadt.de:8080/det/. Dtermined.ai docs are at: https://docs.determined.ai/latest/index.html

## Installation
On your local machine:
1. Install:
```
pip install determined==0.31.0 (Will be updated to 0.35.0 soon)
```
Specifying the determined version is important because the cluster already runs an old version of determined and having a mismatching version between client(your workstation) and the cluster works most of the times but for example when connecting to a shell it does not work.

2. Add the following lines to your `~/.bashrc` or `~/.profile` (or `~/.zshrc` on macOS):
```
export DET_MASTER=https://login01.ai.tu-darmstadt.de:8080
export DET_USER=<USER>
export DET_WSP=<WSP>
export DET_POOL=42_Compute
```
For PEARL, the WSP is PI_Chalvatzaki

3. Run `det user login <USER>`

## Launching Shells
To launch a test shell with the default setup on the cluster, run:
```
det shell start -w ${DET_WSP} --config resources.slots=<NUM_GPUS>
```
The above command will launch a minimal setup with NVIDIA A100 GPUs (if NUM_GPUS>0), python3.8, torch1.12, cuda11.3 etc. *Note that GPU hours start getting billed as soon as the node is created and only stop counting when the node has been killed by explicitly calling the `det shell kill` command!*

## Connect to a Shell
`det shell` shows a list of shells and their corresponding `<id>`.

Connect to a shell using `det shell open <id>`

### Via SSH
1. Run `det shell show_ssh_command <id>`. This will create an output with the following pattern:
```
ssh -o "ProxyCommand=<PROXY>" -o StrictHostKeyChecking=no -tt -o IdentitiesOnly=yes -i <KEY_LOCATION> <USER>@<ID>
```
2. Copy `<KEY_LOCATION>` to a location where you can change the permissions using `chmod` e.g., `/visinf/projects/<username>/determined/key` lets call this `NEW_KEY_LOCATION`
3. `chmod 600 <NEW_KEY_LOCATION>`
4. Edit your `~/.ssh/config` and add the following lines:
```
Host cluster
        HostName <ID>
        ProxyCommand <PROXY>>
        StrictHostKeyChecking no
        IdentitiesOnly yes
        IdentityFile <NEW_KEY_LOCATION>
        User <USER>
```
4. You can now connect to your shell using `ssh cluster` and use rsync as usual `rsync -az <SRC> cluster:<DST>` for data uploading
5. Repeat all steps for every new shell you create

If you followed this, you can also setup your IDE to use the `cluster` hostname for remote development

## Shutting down
`det shell kill <ID>`

To avoid wasted GPU hours I recommend to automatically kill the shell after the training script was executed. You can achieve this by running something like this within your shell:
```
python train.py; det shell kill $DET_TASK_ID
```
This command will run your `train.py` script and afterwards the shell will kill itself. For monitoring (e.g. using Tensorboard) you can start a shell without GPUs and just run your monitoring tool there.

## Launch shells with basic docker config and mounts (minimal setup that is required)
To launch a node with a basic docker config with mounts, replace `<WSP>` and `<USER>` in [pearl_hessian_cluster/basic_config.yaml](pearl_hessian_cluster/basic_config.yaml) by your `DET_USER` and `DET_WSP` and run the following command:
```
det shell start -w ${DET_WSP} --config-file basic_config.yaml --config resources.slots=<NUM_GPUS>
```
This launches a very minimalistic image. A persistent storage on the cluster for the PI workspace is at `/pfss/mlde/workspaces/mlde_wsp_<WSP>` and for the user at `/pfss/mlde/users/<USER>`. Note that this user directory is only to save passwords or user critical information and thus only has 2GB of allocated disk space. You should use the PI workspace that has an allocation of 5 TB. The default setup is hard to use. See below on how to set up a node that uses our custom docker image.

## Using Conda in the Basic Docker Environment

### Environment Creation
Create environments only in your persistent directories mounted within `/pfss/mlde/workspaces/mlde_wsp_<WSP>`. Example:
```
conda create --prefix /pfss/mlde/workspaces/mlde_wsp_<WSP>/conda/test python=3.10
```

### Loading a Conda Environment
For every new shell that you create you have to do the following steps:
1. `conda init bash`
2. `source ~/.bashrc`

Now you can activate your environment with
```
conda activate /pfss/mlde/workspaces/mlde_wsp_<WSP>/conda/test
```

### Additional Thoughts
The default docker image misses a few essential system tools like:
- Apparently there is no editor installed (`vi`, `vim`, `ed`, `emacs`, `nano`). If anyone finds an editor, pls tell me :D
- `unzip`, `zip`, `p7z` or similar
- Really basic commands like `ping` are missing
- No `screen` or `tmux`
- Providing an own docker image to fix this apparently requires to host it publicly on Docker Hub. This means that, depending on the project, our environments and maybe parts of our code need to be publicly available which is not optimal.
- I would not recommend using the `PyTorchTrial` experiments as described in the determined.ai docs, because it really limits you in your logging capabilities and you lose a lot of control over your training pipeline. Further, if you want to analyze the logs that are created automatically, your tensorboard get shutdown after 5 minutes(it's a feature as confirmed by HP)


## Custom docker image based nodes
My goal is to enable launching shells on the cluster that are usable for everyone.

### Initial Configuration
This has to be done only once. Its to set up a shared directory and your own directories in the PI workspace.

In the following, commands executed on your local machine are marked with `#` and commands executed on the cluster are marked with `$`.

1. Launch a shell without GPUs as described in [Launch shells with basic docker config], i.e.
`det shell start -w ${DET_WSP} --config-file basic_config.yaml --config resources.slots=0`
2. `$ cd /pfss/mlde/workspaces/mlde_wsp_<WSP>`
3. If the folders `home`, and `shared` do not exist yet, run `mkdir -m 770 home shared`
4. `$ mkdir -m 700 -p home/<USER>`
5. `$ exit`
6. `# det shell kill <ID>`
7. Replace `<WSP>` and `<USER>` in [pearl_hessian_cluster/advanced_config.yaml](pearl_hessian_cluster/advanced_config.yaml) by your `DET_USER` and `DET_WSP`
8. `# det shell start -w ${DET_WSP} --config-file advanced_config.yaml`. (This may take a while the first time the image is pulled)
9. `$ conda init bash`
10. `$ source ~/.bashrc`
11. `$ conda activate`
12. `$ exit`
13. `# det shell kill <ID>`

After step 11, you should be in a conda environment. If all of this works, the initial configuration is done.

### Usage
Replace `<WSP>` and `<USER>` in [pearl_hessian_cluster/advanced_config.yaml](pearl_hessian_cluster/advanced_config.yaml) by your `DET_USER` and `DET_WSP`.
```
det shell start -w ${DET_WSP} --config-file advanced_config.yaml --config resources.slots=<NUM_GPUs>
```
All persistent folders(`/shared`, and your home directory) are in the PI workspace because workspaces have a bigger fs quota than user folders.

If you need to install additional packages, you can acquire root privileges using `su`. The root password in the docker container is set to `root`.

### Extending the Docker Image
The `Dockerfile` I used to create this environment is at [pearl_hessian_cluster/Dockerfile](pearl_hessian_cluster/Dockerfile). You have to upload your new Docker container to [https://hub.docker.com](https://hub.docker.com) and change the `cpu` and `cuda` images in `advanced_config.yaml` accordingly.

### Launching a determined experiment instead of shell
You can launch a determined experiment with an entrypoint instead of connecting to a shell.

The experiment can be tracked in the determined web browser [https://login01.ai.tu-darmstadt.de:8080/det](https://login01.ai.tu-darmstadt.de:8080/det). Have a look at an example in the `2haff_cluster_exp_config.yaml` file. The full reference for det experiment commands is available here: [https://docs.determined.ai/0.17.2/training-apis/experiment-config.html](https://docs.determined.ai/0.17.2/training-apis/experiment-config.html).

To launch the experiment, run:
```
det experiment create ./2haff_cluster_exp_config.yaml
```

### Untested Features
Here is a list of features that should work but are not tested yet:
- Manuel Brack has created a Docker image that could be useful: `mbrack/forty-two:cuda-12.5-pytorch-2.2-gpu-mpi-multimodal`. It uses torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1, 
already has cuda and cuda-toolkit at 12.5 preinstalled.
Some of the more annoying dependencies like flash-attn and all of rapids are also included.
- [ ] Infiniband. All drivers should be installed and the Infiniband device is mounted at `/dev/infiniband/`

### Bugs / Missing Features
Feel free to add bugs and missing features in the Docker image here.
