# Rubik's Cube Solver

This software does the following:
1. Find the rubik's cube top face by segmenting the input image
2. Get the colors of the face's cells
3. Propose a solution after scanning all the faces

Usually this is done by scanning the complete cube in a specific sequence (Top, Left, Front, Right, Back, Bottom).

The optimized sequence of movements to solve the cube is saved to the `moves.txt` file, and they're overlaid with AR over the cube.

Check out the slides for more details at [`story/slides`](story/slides.md).

## Usage

Initialize the project and install the dependencies:

```sh
py -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`src/main.py` contains an augmented reality program to solve the cube.

`src/rerun_demo.py` contains a rerun environment to preview different aspects of the program, such as:
- Edge detection
- Cube's face segmentation
- Etc

### Flags

You can run both `src/main.py` and `src/rerun_demo.py` with the following flags:
- `-c CAMERA_INDEX`: Run using the `CAMERA_INDEX`th video input as the camera.
- `-g GAMMA`: Pass in a float (usually between 0.6 and 1.4) to specify the gamma correction. 

## Demos

### Rerun
[Visualizing different stages (edges, color extraction, etc) with Rerun](https://youtu.be/PNeJHCyeSPo)

### Augmented Reality
[Storing the cube's faces and displaying the moves with augmented reality](https://youtu.be/L2Ida9gxqpY)
