import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from kaggle_secrets import UserSecretsClient

START = time.time()
COMMIT = "0dd13fd2b48ec021ccf92dc3ffa79ec3bedbe893"
REPO = "https://github.com/Mohidali2005/no-word-for-it.git"
AUTHORS_REPO = "https://github.com/Patchwork53/VLMs-Need-Words-COLM2026.git"
AUTHORS_COMMIT = "18655d407bbe69864eea3d2f13bba0144c8f4e33"
CODE = Path("/tmp/code")
OUT = Path("/kaggle/working")
# attached datasets do not always appear at the same place so the stimuli are found by one of their sets
found = [Path(folder).parent for folder,dirs,files in os.walk("/kaggle/input",followlinks=True) if Path(folder).name=="rep_named" and "items.jsonl" in files]
assert len(found)==1,f"expected one stimuli folder under /kaggle/input but found {found}"
STIMULI = found[0]
# the suite runs first on the preinstalled library and then again on the older one the laptop also uses
VERSIONS = ["preinstalled","4.57.1"]


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
assert shell(f"git -C {CODE} rev-parse HEAD")==COMMIT,"wrong code commit"

# the tests compare against the authors own functions and shapes so only that folder of their public repo is fetched
authors = CODE/"data"/"raw"/"vnw_repo"
subprocess.run(["git","clone","--quiet","--filter=blob:none","--no-checkout",AUTHORS_REPO,str(authors)],check=True)
subprocess.run(["git","-C",str(authors),"sparse-checkout","set","2D_shape_recognition"],check=True)
subprocess.run(["git","-C",str(authors),"checkout","--quiet","--detach",AUTHORS_COMMIT],check=True)
assert shell(f"git -C {authors} rev-parse HEAD")==AUTHORS_COMMIT,"wrong authors commit"
print(f"code at {COMMIT} and authors code at {AUTHORS_COMMIT} and stimuli at {STIMULI}")

env = dict(os.environ,PYTHONPATH=str(CODE/"src"),NWF_STIMULI=str(STIMULI),HF_HOME="/tmp/hf",HF_TOKEN=hf_token,MPLBACKEND="Agg")
# the same test runner version as on the laptop
subprocess.run([sys.executable,"-m","pip","install","--quiet","pytest==9.2.0"],check=True)
summary = []
for version in VERSIONS:
    if version!="preinstalled":
        subprocess.run([sys.executable,"-m","pip","install","--quiet",f"transformers=={version}"],check=True)
    packages = shell("pip list 2>/dev/null | grep -i -E '^(torch|torchvision|transformers|tokenizers|accelerate|safetensors|huggingface.hub|numpy|scipy|pillow|matplotlib|pandas|pyarrow|pytest) '")
    began = time.time()
    with open(OUT/f"tests_{version}.log","w") as log:
        code = subprocess.run([sys.executable,"-m","pytest","tests","-p","no:warnings","-v","--durations=25",f"--junitxml={OUT}/tests_{version}.xml"],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    summary.append({"version":version,"exit_code":code,"minutes":round((time.time()-began)/60,1),"packages":packages})
    (OUT/"tests_summary.json").write_text(json.dumps({"commit":COMMIT,"python":sys.version,"runs":summary},indent=1),encoding="utf-8")
    print(f"tests on {version} ended with code {code} after {summary[-1]['minutes']} min")
print(f"finished after {(time.time()-START)/3600:.2f} h")
