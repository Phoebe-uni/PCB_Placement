# PCB_Placement

This project is about using PPO, A2C and DDPG algorithms to realize PCB auto placement. Part of code is referenced to https://github.com/lukevassallo/rl_pcb.git.

## Code Structure

- Folder dataset includes the datasets used in the project. 
- Folder defaults includes the .json files which contains the hyperparameters used in each algorithms
- In the experiments folder, 15_226, 15_262, 15_442 and 15_622 includes the code used in the experiments with 4 sets of configurations. 
- Folder src includes the code used for training and generating evaluation reports. Folder training includes the code of reinforcement algorithms, data processing and all the code used for construct the model. Other 2 folders includes the code for generating reports.
- Folder test includes the code used for testing different algorithms.



## Run automated installation script
The installation of this projet is referenced to https://github.com/lukevassallo/rl_pcb.git. 
The automated installation procedure makes the following changes to the local repository:
- Create a directory bin and installs the KiCad parsing utility, and place and route tools
- Creates an environment using python3.8, installs pytorch 1.13 with CUDA 11.7 and all necessary python packages
- Installs the wheel libraries in the lib folder

```
./install_tools_and_virtual_environment.sh --env_only
```

If you do not have CUDA 11.7 installed, you can install the CPU only. Tests an experiments will run significantly slower but works out of the box.
```
./install_tools_and_virtual_environment.sh --env_only --cpu_only
```

If you require a different version of CUDA, please make the following changes and run `install_tools_and_virtual_environment.sh` without any options:
- To `setup.sh` script, change the CUDA path to point to your installation of CUDA.
- To `install_tools_and_virtual_environment.sh` script, change the PyTorch libraries to use your CUDA version. 

Using a CPU device or alternative CUDA version will yield different results than the accompanying pdf reports for tests and experiments.

# Run tests and experiments
Always source the environment setup script before running any tests or experiments. **The script should be run from the root of the repository**
```
cd <path-to-rl_pcb>
source setup.sh 
```

Run an experiment
```
cd experiments/00_parameter_exeperiments
./run.sh    
```

Run a test - tests are used to validate the correct operation of the source code. They are periodically run in a Continuous Integration (CI) environment.
```
cd tests/01_training_td3_cpu
./run.sh
```

The script `run.sh` will perform the following: 
1. Carry out the training run(s) by following the instructions in `run_config.txt` that is located within the same directory
2. Generates an experiment report that processes the experimental data and presents the results in tables and figures. All experiment metadata is also reported, and customisation is possible through `report_config.py`, located within the same directory.
3. Evaluate all policies alongside simulated annealing baseline. All optimised placements are subsequently routed using an A\* based algorithm. 
4. Generate a report that processes all evaluation data and tabulates HPWL and routed wirelength metrics. All experiment metadata is also reported.

The generated files can be cleaned by running the following:
```
./clean.sh
```

Every test and experiment contains a directory called `expected results` that contains pre-generated reports. Should you run the experiments as provided, identical results are to be expected.

# GPU Setup (Optional)
This section provides an optional setup procedure to remove the Nvidia GPU driver and all dependent libraries and perform a fresh install. **The commands in this section make big changes to your system. Please read carefully before running commands** Some command changes will be required.

1. Remove CUDA if installed `sudo apt-get --purge remove cuda* *cublas* *cufft* *curand* *cusolver* *cusparse* *npp* *nvjpeg* *nsight*`
2. Check if the driver is installed, and if it is, remove it with `sudo apt-get --purge remove *nvidia*`
3. Remove cuddnn `sudo apt remove libcudnn* libcudnn*-dev`

### Install Nvidia GPU Driver
To install the driver, you can start by issuing the command `ubuntu-drivers devices` and identifying the latest third-party non-free version. Once you have identified the appropriate version, use `apt` to install the driver. After installing the driver, reboot your system and issue the command `nvidia-smi` to identify the full driver version. In the upcoming section, you will need this information to determine which CUDA version is supported.

### Download and install CUDA toolkit
To ensure that your device driver is compatible with CUDA, you'll need to check the compatibility using the following link: https://docs.nvidia.com/deploy/cuda-compatibility/. Once you've confirmed the compatibility, you can proceed to the CUDA Toolkit Archive at https://developer.nvidia.com/cuda-toolkit-archive. From there, select version 11.7 and choose the appropriate platform parameters from the "Select Target Platform" section. Next, download the runfile (local) and proceed with the installation process. Finally, follow the installation instructions carefully and avoid installing the driver when prompted.

```
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run
```

Update the setup.sh script as necessary. The default contents for `PATH` and `LD_LIBRARY_PATH` are:

```
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```
