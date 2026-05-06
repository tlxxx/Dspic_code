The Code for the **D**iffusion **S**oft **P**olicy **I**teration with **C**omplete Division (Dspic) paper submission at ICML2026. (accepted as a regular paper)

## Towards Complete Multi-Agent Coordination Policy Learning via Denoising Maximum Entropy Optimization 
**Learning Curves are available in *paper_plots***


### Installation
Ensure your Python version >= 3.11, then you can install our repository by:

```bash
pip install -r requirement.txt
```
To install SMAC, please follow the official instructions in [here](https://github.com/oxwhirl/smac).
To install SMACv2, please follow the official instructions in [here](https://github.com/oxwhirl/smacv2). 
To install LBF, please follow the official instructions in [here](https://github.com/semitable/lb-foraging).
To install MaMuJoCo, please follow the instructions on https://github.com/openai/mujoco-py, https://www.roboti.us/, and https://github.com/deepmind/mujoco to download the right version of mujoco you need (mujoco210 is suggested).  

Then, ```mkdir ~/.mujoco``` and move the .tar.gz or .zip to ```~/.mujoco```, and extract it by ```unzip zipname```. Finally add the path to ```~/.bashrc``` with 

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/<folder-name/bin
```

After installation is finished, the conda environment can be activated, and the code can be run using 

```python
python examples/train.py
```

### Running

You can modify the corresponding algorithm and environment parameters in ```src/configs```, and our paper also provides information on the parameters we use.

You can freely choose the algorithm to run (currently only dspic is supported), the testing environment, and the experiment name, simply by running with

```python
python examples/train.py --algo dspic --env smac/smacv2/mamujoco/lbf --exp_name test1
```


### Acknowledgements
Portions of the project are adapted from other repositories: 
- https://github.com/PKU-MARL/HARL is licensed under MIT,
- https://github.com/ALRhub/DIME is licensed under MIT. 
