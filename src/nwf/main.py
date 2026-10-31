import argparse
import hashlib
import json
import subprocess
import sys
import time

from nwf import config
from nwf import precision
from nwf.paths import RESULTS
from nwf.paths import STIMULI
from nwf.paths import STIMULI_MANIFEST

# the smoke run scores every model twice so the precision check can compare full fp32 with the fast mode
SMOKE_PLAN = ["qwen:full","qwen:fast","qwen_names:full","qwen_names:fast","gemma:full","gemma:fast"]


def check_stimuli():
    """Stop unless every stimulus set on disk has exactly the hashes of the local build"""
    expected = json.loads(STIMULI_MANIFEST.read_text(encoding="utf-8"))
    for name,info in expected["sets"].items():
        files = sorted(p for p in (STIMULI/name).rglob("*") if p.is_file())
        listing = "".join(f"{p.relative_to(STIMULI).as_posix()} {hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in files)
        assert hashlib.sha256(listing.encode()).hexdigest()==info["all_files_sha256"],f"{name} stimuli differ from the local build"
    print(f"all {len(expected['sets'])} stimulus sets match the local build")


def kaggle(plan,smoke,stop_at):
    """Run each model of the plan in its own process and record how every step ended"""
    check_stimuli()
    report = []
    steps = []
    for entry in plan:
        name,mode,*sets = entry.split(":")
        command = ["--model",name,"--mode",mode,"--stop-at",str(stop_at)]+(["--smoke"] if smoke else [])
        # an optional fourth field keeps only the first items of each set
        steps.append((entry,["nwf.run_items",*command]+(["--sets",sets[0].replace("+",",")] if sets else [])+(["--items",sets[1]] if len(sets)>1 else [])))
        # the removal check runs in its own process after the model has been scored on every set
        if name in config.REMOVAL_MODELS and not sets:
            steps.append((f"{entry}:removal",["nwf.removal",*command]))
    for step,module in steps:
        if time.time()>stop_at:
            report.append({"step":step,"skipped":"deadline"})
            continue
        began = time.time()
        code = subprocess.run([sys.executable,"-m",*module]).returncode
        report.append({"step":step,"exit_code":code,"minutes":round((time.time()-began)/60,1)})
        RESULTS.mkdir(parents=True,exist_ok=True)
        (RESULTS/"run_report.json").write_text(json.dumps(report,indent=1),encoding="utf-8")
        print(f"step {step} ended with code {code} after {report[-1]['minutes']} min")
    if smoke:
        done = [e["step"] for e in report if e.get("exit_code")==0]
        for name in dict.fromkeys(e.split(":")[0] for e in done):
            if f"{name}:full" in done and f"{name}:fast" in done:
                precision.compare(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run the study stages")
    parser.add_argument("--kaggle",action="store_true",help="run the gpu stages")
    parser.add_argument("--smoke",action="store_true",help="the short gpu run that checks every code path and the precision")
    parser.add_argument("--plan",help="comma separated model:mode or model:mode:set+set or model:mode:set+set:items entries for the main gpu run")
    parser.add_argument("--stop-at",type=float,default=float("inf"),help="unix time after which no new work starts")
    args = parser.parse_args()
    if args.kaggle:
        kaggle(SMOKE_PLAN if args.smoke else args.plan.split(","),args.smoke,args.stop_at)
    else:
        # the local stages so far only confirm the stimuli and later analysis stages are added here
        check_stimuli()
