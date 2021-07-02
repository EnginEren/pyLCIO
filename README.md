
### Local 

```bash
docker run -it --rm -v $PWD/pyLCIO/examples:/home/ilc/data -p 8888:8888 ilcsoft/py3lcio:lcio-16 bash
```

this will spin up a container with port mappings to your local PC as well as a local path mounted on to the container.

```bash
cd LCIO; source setup.sh; cd .. 
conda activate root_env ## if you wish to use matplotlib instead of ROOT
jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root 
```

Click the link from your screen which starts with "http://127.0.0.1:8888/?token=XXxXXXXX...". Then you should see the notebook. You create a notebook from the upper-right cornet with 'python 3' kernel and then type `import pyLCIO`


### NAF

You can run `ilcsoft/py3lcio:lcio-15-04` in NAF via `singularity`. But first it's practical to do

```bash
export SINGULARITY_TMPDIR=/nfs/dust/ilc/user/<name>/container/tmp/
export SINGULARITY_CACHEDIR=/nfs/dust/ilc/user/<name>/container/cache/
```
then

```bash
git clone https://github.com/EnginEren/pyLCIO.git
cd pyLCIO
singularity shell -H $PWD --bind $(pwd):/home/ilc/data docker://ilcsoft/py3lcio:lcio-15-04 bash
cd /home/ilc/LCIO; source setup.sh; cd .. 
conda activate root_env 
```
*note*: if you are activating conda env, it will ask you to initilize the shell. `conda init bash`. This is something you do it once.

```bash
jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root 
```
If you're not in DESY network, you need to tunnel via ssh. Follow official DESY-IT page [here](https://it.desy.de/services/uco/documentation/external_access/index_eng.html).

After that, you can copy notebook link to your browser and access it. Well done!



