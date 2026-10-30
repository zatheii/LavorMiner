# LavorMiner

A Windows bot that finds blue crystal nodes on screen in a Roblox snow field, walks to the nearest one, mines it, and repeats. It steers by sight using OpenCV colour detection, so it needs no route or compass.

Automating gameplay can break Roblox's rules. Use it at your own risk.

## Setup

1. Install Python 3 for Windows (the `py` launcher).
2. Install the dependencies:

   ```
   py -m pip install -r requirements.txt
   ```

3. Open a console **as Administrator** so clicks and key presses reach Roblox.

## Usage

```
py crystal_miner_v18_menu.py         # open the menu (start here)
py crystal_miner_v18_menu.py run     # live dashboard: F6 = start/stop, Q = quit
py crystal_miner_v18_menu.py test    # save an annotated screenshot of what it detects
py crystal_miner_v18_menu.py calib   # live HSV readout under the cursor
```

It works best at dusk in-game, when the crystals stand out clearly from the snow. Tuning values (colour thresholds, distances, steering) are in the `SETTINGS` block at the top of the script, and some can be changed from the menu.

## Building the .exe

Run `build_exe.bat`. It creates a throwaway virtual environment in `%TEMP%`, installs PyInstaller and the dependencies there, and writes `CrystalMiner.exe` next to the script. Only the PC doing the build needs Python.

## Runtime files

The miner writes `miner_log.txt`, `debug_view.png`, and `crystal_detect.png` next to the script. They are ignored by git.
