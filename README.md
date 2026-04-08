# Motion Estimation

## Overview

**Motion Estimation** is a fundamental technique used in video processing and compression that aims to find the motion vectors representing the movement of objects between consecutive frames in a video sequence. It plays a crucial role in video coding standards as H.264
The primary goal of motion estimation is to describe the spatial transformation that maps pixels from one frame to another, enabling efficient compression by exploiting temporal redundancy in video.

## Motion Vector

A **Motion Vector** is a 2D displacement vector that represents the movement of a macroblock (a rectangular region of pixels) from its position in one frame to another. Motion vectors are characterized by:

- **Horizontal Component (dx)**: The displacement in the x-axis (left-right direction)
- **Vertical Component (dy)**: The displacement in the y-axis (up-down direction)

Motion vectors are typically calculated by finding the best match between a block in the current frame and candidate blocks in a reference frame, minimizing a cost function such as the Sum of Absolute Differences (SAD).

