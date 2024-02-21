import os
import shutil
import time

with open("config.yaml") as f:
    amorph_config = f.read()

#bands = []
#with open("torun.txt") as f:
#    for line in f:
#        files.append(f)


def write_torun(files):
    with open("torun.txt") as f:
        for f in files:
            f.write(f"{f}\n")

dirname = "data"
files = os.listdir(dirname)
os.makedirs("runs", exist_ok=True)

for i, b in enumerate(files):
    name = os.path.splitext(b)[0]
    print(f"\nProcessing file '{name}'")
    start_time = time.monotonic()

    rundir = f"runs/{name}"

    os.makedirs(rundir, exist_ok=True)
    shutil.copyfile("OPTIONS", f"{rundir}/OPTIONS")
    shutil.copyfile(f"{dirname}/{b}", f"{rundir}/{b}")
    with open(f"{rundir}/config.yaml", "w") as f:
        f.write(amorph_config.replace("$FILENAME", b))
    
    os.chdir(rundir) 
    os.system("AMORPH")
    os.chdir("../..")

    end_time = time.monotonic()
    print(f"\tTook {end_time - start_time} s")

    if i == len(files) - 1:
        continue


