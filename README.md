# Real Time Spike Sorting using Kilosort4 and Bonsai
This is a slight repackage of my thesis project that aims to reduce bloat (at least a little bit) and make it easier for someone else to carry on where I left off.
This project aims to implement real-time template matching by first collecting data and running kilosort4 on it to extract templates, then using the Bonsai suite with some custom functions to load and match templates. Full implementation details can be found in thesis.pdf

# Environment Setup
## Kilosort
The original Kilosort4 (found [here](https://github.com/MouseLand/Kilosort)) is pretty good for standard things (ie spike sorting unlabelled data). If you want to do full validation, I made a fork of their repo which I'm happy to share on request, but it might also just be out of date by now.
Note that your computer should have CUDA compatibility to run Kilosort4

## Bonsai
Install the following libraries (dummy data only):
- OpenTK
- OpenEphys.onix1
- Bonsai.IO (Bonsai - System Library)
- Bonsai DSP
- Bonsai Visualizers
- Bonsai - scripting library
- Bonsai - DSP design library

Note that if you want to edit any scripts, bonsai runs on C# 4.7.2

## Configuration
Go to src/launch_gui.py and check all the strings are appropriate (they won't be to start with, they're all hardcoded to my old laptop)
Go into bonsai and check the parameters at each node are also appropriate

# Running
Run launch_gui.py to get a stepthrough of this process.
The general pipeline works like this:
1. Collect data and write to .bin file
2. Run data through kilosort to extract templates
3. (optional) use [Phy](https://github.com/cortex-lab/phy) to select target units
4. Run Bonsai

# Further Work
If you're looking at this you probably have your own ideas, but the major ones I'd recommend:
- Implement data acquisition and actual real-time pipeline (there's already a bonsai library to do this, ask me or david. For proof of concept reasons, my thesis worked exclusively on dummy data, but the infrastructure exists, thanks david)
- Swap cosine similarity for something like euclidean distance and see what that does

# TODO (RA)
Run through the pipeline on a fresh computer and see what the actual setup / run steps are.