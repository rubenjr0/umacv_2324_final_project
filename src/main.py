import rerun as rr
from vision import _load_image, _fix_perspective, _segment_colors, _get_face_colors
from solver import solve

rr.init("rerun_example_demo", spawn=True)


sample = _load_image("data/1.png")
fixed, M = _fix_perspective(sample)
segmented = _segment_colors(fixed, boost=True)
colors, centers = _get_face_colors(segmented)
action = solve(colors)
print(M)
print(colors)
print(action)


rr.log("image", rr.Image(sample))
rr.log("transformed", rr.Image(fixed))
rr.log("transformed/segmented", rr.Image(segmented))
