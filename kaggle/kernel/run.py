import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kaggle_secrets import UserSecretsClient

START = time.time()
# smoke checks every code path and the precision of each model while main scores the plan below
MODE = "main"
COMMIT = "d96d7ddbc6b99791a352db5beae5900796f2a65b"
REPO = "https://github.com/Mohidali2005/no-word-for-it.git"
# model and precision pairs for a main run as passed by the smoke run for example qwen:fast,qwen_names:fast
PLAN = "qwen:fast,qwen_names:fast"
CODE = Path("/tmp/code")
RESULTS = Path("/kaggle/working/results")
# attached datasets do not always appear at the same place so the stimuli are found by one of their sets
found = [Path(folder).parent for folder,dirs,files in os.walk("/kaggle/input",followlinks=True) if Path(folder).name=="rep_named" and "items.jsonl" in files]
assert len(found)==1,f"expected one stimuli folder under /kaggle/input but found {found}"
STIMULI = found[0]
# a results folder of an earlier main run attached as a dataset lets a main run continue from it
RESUME = [Path(folder) for folder,dirs,files in os.walk("/kaggle/input",followlinks=True) if "environment_main.json" in files]
assert len(RESUME)<=1,f"expected at most one earlier results folder under /kaggle/input but found {RESUME}"
# no new work starts after this many hours so the session still ends inside the twelve hour limit
HOURS = 11.0


def shell(command):
    """Run a shell command and return its output as text"""
    return subprocess.run(command,shell=True,capture_output=True,text=True,executable="/bin/bash").stdout.strip()


secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
github_token = secrets.get_secret("GITHUB_TOKEN")

# passing the github token through git config variables so it never shows up in a command line or a file
auth = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
git_env = dict(os.environ,GIT_CONFIG_COUNT="1",GIT_CONFIG_KEY_0="http.extraHeader",GIT_CONFIG_VALUE_0=f"Authorization: Basic {auth}")
subprocess.run(["git","clone","--quiet",REPO,str(CODE)],env=git_env,check=True)
subprocess.run(["git","-C",str(CODE),"checkout","--quiet","--detach",COMMIT],check=True)
head = subprocess.run(["git","-C",str(CODE),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
assert head==COMMIT,f"checked out {head} instead of {COMMIT}"
print(f"code at commit {head} and stimuli at {STIMULI}")

RESULTS.mkdir(parents=True,exist_ok=True)
if MODE=="main" and RESUME:
    shutil.copytree(RESUME[0],RESULTS,dirs_exist_ok=True)
    print(f"continuing from the earlier results folder {RESUME[0]}")
(RESULTS/f"environment_{MODE}.json").write_text(json.dumps({
    "mode":MODE,
    "commit":COMMIT,
    "plan":PLAN,
    "started":time.strftime("%Y-%m-%d %H:%M:%S"),
    "gpus":shell("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    "packages":shell("pip list 2>/dev/null | grep -i -E '^(torch|torchvision|transformers|tokenizers|accelerate|safetensors|huggingface.hub|numpy|scipy|pillow|matplotlib) '"),
    "python":sys.version,
    "free_disk_gb":round(shutil.disk_usage("/tmp").free/1e9),
},indent=1),encoding="utf-8")

env = dict(os.environ,PYTHONPATH=str(CODE/"src"),NWF_STIMULI=str(STIMULI),NWF_RESULTS=str(RESULTS),NWF_SCRATCH="/tmp/nwf_scratch",HF_HOME="/tmp/hf",HF_TOKEN=hf_token)
args = ["--smoke"] if MODE=="smoke" else ["--plan",PLAN]
subprocess.run([sys.executable,"-m","nwf.main","--kaggle",*args,"--stop-at",str(START+HOURS*3600)],env=env,check=True)
print(f"finished after {(time.time()-START)/3600:.2f} h")
