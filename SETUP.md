# LogiEdge — Setup Guide for Team Members

**BITS Pilani WILP · AIML ZG535 Machine Learning on Edge · Group 18**

Everything you need to go from a clean Windows machine to a working LogiEdge
pipeline. Allow about **45 minutes**, most of it waiting on downloads.

Every gotcha listed here is one we actually hit during development, so read the
notes rather than skipping them.

---

## 0 · What you need

- Windows 10 (version 2004 / build 19041 or higher) or Windows 11
- ~10 GB free disk
- Admin rights on the machine

Check your Windows build first — press `Win+R`, type `winver`, press Enter.
Below 19041, run Windows Update before going any further; nothing else will work.

---

## 1 · WSL2 (≈15 min, includes a reboot)

**All of this runs in PowerShell as Administrator.**

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot when prompted. Ubuntu opens on restart and asks you to create a username
and password — **remember the password**, it is what `sudo` will ask for.

Verify:

```powershell
wsl -l -v
```

You want `VERSION 2`, not 1.

> **If `wsl --install` is not recognised** (older Windows 10 builds):
> ```powershell
> dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
> dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
> ```
> Reboot, install the [WSL2 kernel update](https://aka.ms/wsl2kernel), run
> `wsl --set-default-version 2`, then install Ubuntu 22.04 from the Microsoft Store.

### From here on, every command goes in the Ubuntu terminal

Not PowerShell. This matters: **`sudo` does not exist in PowerShell or Command
Prompt.** If you see `sudo : The term 'sudo' is not recognized`, you are in the
wrong window.

---

## 2 · Docker

Two routes. Pick one.

### Route A — Docker Desktop (simplest if it installs)

Install Docker Desktop for Windows, then open
**Settings → Resources → WSL Integration** and enable the toggle for
`Ubuntu-22.04`. Apply and restart.

Test from Ubuntu:

```bash
docker ps
```

A table (even empty) means you are done.

> **Do not run `sudo service docker start` on this route.** The daemon lives on
> the Windows side, so there is no `docker.service` inside Ubuntu and the command
> fails with *"Unit docker.service not found"*. That error is expected and
> harmless — just make sure Docker Desktop is running on Windows.

### Route B — Docker Engine inside WSL

Use this if Docker Desktop refuses to install (it may on Windows 10, which is
past end of support).

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo usermod -aG docker $USER
```

**Close the Ubuntu window and reopen it** — group membership only applies to a
new session. Then:

```bash
sudo service docker start
docker run --rm hello-world
```

On this route the daemon does **not** auto-start. Either run
`sudo service docker start` each session, or add it to your shell startup:

```bash
echo 'sudo service docker start > /dev/null 2>&1' >> ~/.bashrc
```

---

## 3 · System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev \
                    mosquitto mosquitto-clients make zip unzip dos2unix git
```

---

## 4 · Get the repository

```bash
cd ~
git clone <REPO-URL> logibridge
cd logibridge
find . -name "*.sh" -exec dos2unix {} \;
chmod +x *.sh demo/*.sh
```

> **Clone into `~`, never into `/mnt/c/`.** Docker builds and file I/O across
> the Windows filesystem boundary are several times slower. Everything works
> from the Linux home directory.

> **The `dos2unix` line is not optional** if the code reached you through
> Windows (a zip, Teams, email). Windows line endings make every `.sh` file fail
> with `$'\r': command not found`.

If you are working from a zip rather than git:

```bash
cd ~
cp /mnt/c/Users/<YourWindowsName>/Downloads/logibridge.zip .
unzip logibridge.zip && cd logibridge
find . -name "*.sh" -exec dos2unix {} \; && chmod +x *.sh demo/*.sh
```

---

## 5 · Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install ansible docker
ansible-galaxy collection install community.docker
```

> **`pip install docker` is easy to miss.** The `community.docker` Ansible
> collection fails without that Python package, and the error it gives does not
> say so clearly.

Remember to `source .venv/bin/activate` in every new terminal.

---

## 6 · Local Docker registry

The Ansible playbook pulls from `localhost:5000`, so the registry must exist
before `make deploy` will work.

```bash
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

---

## 7 · Build everything

```bash
make all
```

Takes a few minutes. You are looking for:

```
GATE PASSED  94.74% > 88%  (blocked split 100.00%)
Class 2 Critical recall: 100.0% random, 100.0% blocked
```

Your exact figures will differ slightly — latency and energy are
hardware-dependent. That is expected; cite your own numbers.

Then build and publish the container:

```bash
make docker
docker tag logibridge-inference:latest localhost:5000/logibridge-inference:latest
docker push localhost:5000/logibridge-inference:latest
```

---

## 8 · Verify

```bash
./demo_run.sh reset
```

This stops any stray broker, pins the M2 model and rebuilds its PSI reference,
clears previous deployment state, and runs a full pre-flight check. Every line
should read `OK`. Fix anything marked `FIX` and run it again.

---

## 9 · Running the demonstration

Two terminals for the broker, one for everything else. All three need
`cd ~/logibridge && source .venv/bin/activate` first.

```bash
# Terminal 1 — leave running for the whole session
mosquitto -v

# Terminal 2
mosquitto_sub -t 'logibridge/#' -v

# Terminal 3 — the stages, in this order
./demo_run.sh sensors     # live pipeline + 3-sigma experiment
./demo_run.sh training    # the 88% gate
./demo_run.sh drift       # PSI crosses 0.25, then recovers
./demo_run.sh deploy      # Ansible changed=N then changed=0
./demo_run.sh ota         # Docker layer cache + bandwidth saving
```

**The order matters.** `ota` finishes with the M3 model deployed, while the PSI
reference is built on M2 — so running `drift` afterwards is refused by a guard.
Always do drift first. If you need to re-run drift later, `./demo_run.sh reset`
puts M2 back.

Run `./demo_run.sh` with no arguments for the full list.

---

## 10 · Finding your files from Windows

The repo lives inside WSL, not on C:. To open it in Windows Explorer:

```bash
cd ~/logibridge
explorer.exe .
```

Or paste this into the Explorer address bar:

```
\\wsl.localhost\Ubuntu-22.04\home\<your-username>\logibridge
```

To copy the figures out for a report:

```bash
cp optimisation/results/pareto_chart.png \
   scenario_architecture/system_architecture.png \
   /mnt/c/Users/<YourWindowsName>/Desktop/
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `sudo: command not found` | You are in PowerShell. Switch to the Ubuntu terminal. |
| `Unit docker.service not found` | Docker Desktop route — skip that command, just make sure Docker Desktop is running on Windows. |
| `Cannot connect to the Docker daemon` | Docker Desktop not running, or on Route B run `sudo service docker start`. |
| `$'\r': command not found` | Windows line endings. `dos2unix <file>` |
| `mosquitto -v` → `Address already in use` | The background service holds port 1883. `sudo service mosquitto stop` then `sudo pkill -x mosquitto` |
| `make deploy` task 5 fails on `localhost:5000` | Registry missing — go back to step 6. |
| `make deploy` task 1 → `Permission denied: /opt/logibridge` | Playbook is missing `become: true`. |
| `make drift` → `MISMATCH: reference built on model …` | The served model differs from the PSI reference. `./demo_run.sh reset` |
| OTA demo shows every layer `CACHED` | BuildKit reused an old chain. `./demo_run.sh ota` handles this — it prunes the cache first. |
| Ansible can't find `community.docker` | `pip install docker` then `ansible-galaxy collection install community.docker` |
| Everything slow | Repo is probably on `/mnt/c/`. Move it to `~`. |

---

## What each stage proves

| Stage | Task | What to look for |
|---|---|---|
| `sensors` | C1, C2, C3 | Vibration ramps 0.45 → 1.2 g gradually; Warning recall falls to 70.6% under corrupted stats |
| `training` | D1 | `GATE PASSED`, Class 2 recall 100% |
| `drift` | E1 | PSI ~0.02 clean → peak 13.9 → recovered below 0.10 |
| `deploy` | E2 | Run 1 `changed=4`, run 2 **`changed=0`** |
| `ota` | D2 | `pip install` CACHED, `COPY model.tflite` rebuilt |

---

## Repository layout

See `README.md` for the full tree and the design decisions behind each
component. `reports/REPORT_OUTLINE.md` maps every measured number to the report
section that needs it.
