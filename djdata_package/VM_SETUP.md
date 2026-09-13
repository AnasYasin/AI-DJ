# VM setup instructions for Claude Code (djdata tier-1 run)

You are setting up a fresh AWS instance (c6i.4xlarge, Ubuntu 24.04, 16 vCPU, 32 GB, gp3 200 GB) to
run the `djdata` seam pipeline. Follow these steps in order. Do not skip a check. Do not start
`djdata run` yourself; Anas starts it in tmux once you report that every check passed.
Report every failure the moment it happens, with the exact error text, and stop at that step.

## 1. System packages

```
sudo apt-get update
sudo apt-get install -y ffmpeg xvfb git tmux unzip aria2
ffmpeg -version | head -1        # must print a version
```

## 2. Conda and the environment

```
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda create -y -n aidj python=3.10
conda activate aidj
python --version                  # must print 3.10.x
```

Everything below runs inside `conda activate aidj`.

## 3. Repository and package

The repo is already cloned by Anas at `~/AI-DJ` (if not: `git clone -b main git@github.com:AnasYasin/AI-DJ.git ~/AI-DJ`).

```
cd ~/AI-DJ
pip install -e ./djdata_package
pip install pytest "yt-dlp[default]"          # [default] adds the JS challenge solver (yt-dlp-ejs)
curl -fsSL https://deno.land/install.sh | DENO_INSTALL=$HOME/.deno sh
echo 'export PATH=$HOME/.deno/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
deno --version                    # must print a version; without it YouTube returns no formats
djdata --help                     # must list the subcommands
```

## 3b. YouTube cookies (required: datacenter IPs must be logged in)

Anas exports a cookie file from an incognito YouTube login on his laptop and copies it to
`~/AI-DJ/yt-cookies.txt` (`scp file aidj:~/AI-DJ/yt-cookies.txt`). Then:

```
chmod 600 ~/AI-DJ/yt-cookies.txt
```

and in `djdata_package/config.yaml` set `download.cookies_file: /home/ubuntu/AI-DJ/yt-cookies.txt`.
Leave `youtube_player_client` at `default,web_embedded`; the default client fails with cookies
("The page needs to be reloaded", yt-dlp issue 17389). Never log in to that Google account elsewhere
while the run uses the file.

Do NOT install the repository's requirements.txt. The pipeline needs only the package's own
dependencies. Installing torch, essentia and tensorflow here wastes an hour and can fail.

## 4. Raveform dataset (1.4 GB unzipped, CC BY 4.0)

```
mkdir -p data/raw/raveform && cd data/raw/raveform
aria2c -x 8 -s 8 -o raveform.zip "https://huggingface.co/datasets/taejunkim/raveform/resolve/main/raveform.zip"
unzip -q raveform.zip && rm raveform.zip
ls raveform                       # must show: alignments beats mixes.jsonl mixesdb_genres.jsonl structures tracks.jsonl
wc -l raveform/mixes.jsonl        # must print 4911
cd ~/AI-DJ
```

## 5. Config for this machine

Edit `djdata_package/config.yaml`:

- `workers.seams: 8` (16 vCPU box; laptop default is 4)
- leave `workers.mix: 1`, `workers.tracks: 4`, `run_tiers: [tier1]` as they are
- all paths are relative, so every `djdata` command below must run from `~/AI-DJ`

## 6. Checks, in this order

```
cd ~/AI-DJ
python -m pytest djdata_package/tests -q          # must print "4 passed"
djdata manifest --config djdata_package/config.yaml
```

The manifest must print `"seams": 14080, "mixes": 2084` with tier1 1474 and tier2 12606.
Then:

```
djdata status --config djdata_package/config.yaml # seams pending 1474
djdata probe --config djdata_package/config.yaml  # 20 YouTube track downloads
```

The probe is the one check that can fail for reasons outside the code. Report its summary line
(`ok`, `failed`, `seconds`, `errors`) verbatim.

- 18 or more ok: YouTube serves this machine. Continue.
- `Sign in to confirm you're not a bot`, HTTP 403/429, or `The page needs to be reloaded`: the cookie step (3b) is missing or wrong.
  Stop and report. Do not retry in a loop, that makes the block worse. Anas decides between a
  cookies file (`download.cookies_file` in config.yaml) and fetching tracks from his laptop.
- A handful of `Video unavailable`: normal, those tracks are gone from YouTube. Continue.

## 7. One-mix smoke test

```
sed 's#^root: data/djdata#root: data/djdata_smoke#' djdata_package/config.yaml > /tmp/smoke.yaml
djdata manifest --config /tmp/smoke.yaml
python - <<'EOF'
import sqlite3
c = sqlite3.connect("data/djdata_smoke/state.sqlite")
c.execute("DELETE FROM seams WHERE mix_id!='mix4966'"); c.execute("DELETE FROM mixes WHERE mix_id!='mix4966'")
c.execute("DELETE FROM tracks WHERE track_id NOT IN (SELECT a FROM seams UNION SELECT b FROM seams)"); c.commit()
EOF
timeout 900 djdata run --config /tmp/smoke.yaml
djdata export --config /tmp/smoke.yaml
```

Expected: the run ends by itself within about 3 minutes with `seams: done 6, failed 1` (the one
failure is a track that YouTube no longer serves, its error says so). `data/djdata_smoke/out/seams.csv`
must have 6 rows and the row for seam `mix4966_0OVwkwBsYXM_MCjokw9pMgQ` must have `bass_swap_t`
within 1 second of 4408.0. `data/djdata_smoke/mixes_tmp/` must be empty (full mixes are deleted).
Then `rm -rf data/djdata_smoke`.

## 8. Report

Send Anas one message with: the probe summary line, the smoke-test counts and the bass swap time,
`df -h /` free space, and the exact command to start the real run:

```
tmux new -s djdata
cd ~/AI-DJ && conda activate aidj
djdata run --config djdata_package/config.yaml 2>&1 | tee -a data/djdata/logs/console.log
```

Detach with Ctrl-b d. Progress prints every minute; `djdata status --config djdata_package/config.yaml`
works from any other shell. A killed run resumes with the same command.

## While it runs (for later, not for setup)

- `djdata archivable --config djdata_package/config.yaml` lists track and window files no unfinished
  seam still needs. After moving them to S3, record it: `djdata archived --config ... <files>`.
- `djdata export --config djdata_package/config.yaml` can run at any time; it reads finished seams only.
- Tier 2 is a config change (`run_tiers: [tier1, tier2]`) and the same run command.
