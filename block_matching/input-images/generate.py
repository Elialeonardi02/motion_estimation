#!/usr/bin/env python3

import os
import math


def diagonal_pattern(size):
    pixels = []
    period = size // 4

    for i in range(size):
        for j in range(size):
            val = 128 if ((i + j) // period) % 2 == 0 else 64
            pixels.append(val)

    return pixels


def gradient_pattern(size):
    pixels = []

    for i in range(size):
        for j in range(size):
            val = int(255 * (i / size))
            pixels.append(val)

    return pixels


def checkerboard_pattern(size):
    pixels = []
    square_size = size // 8

    for i in range(size):
        for j in range(size):
            val = 200 if ((i // square_size) + (j // square_size)) % 2 == 0 else 50
            pixels.append(val)

    return pixels


def circles_pattern(size):
    pixels = []
    center = size // 2
    step = size // 8

    for i in range(size):
        for j in range(size):
            dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
            circle_idx = int(dist / step)
            val = 100 if circle_idx % 2 == 0 else 200
            pixels.append(val)

    return pixels


PATTERNS = {
    '1': {'name': 'diagonal', 'description': 'Diagonal pattern', 'func': diagonal_pattern},
    '2': {'name': 'gradient', 'description': 'Gradient pattern', 'func': gradient_pattern},
    '3': {'name': 'checkerboard', 'description': 'Checkerboard pattern', 'func': checkerboard_pattern},
    '4': {'name': 'circles', 'description': 'Concentric circles', 'func': circles_pattern},
}

RESOLUTIONS = ['256', '512', '1024', '2048', '4096', '8192']


def get_user_input():
    print("\n" + "=" * 60)
    print("High-Resolution Square Frame Generator")
    print("=" * 60)

    for i, res in enumerate(RESOLUTIONS, 1):
        print(f"  {i}. {res}x{res}")

    while True:
        try:
            res_choice = input(f"\nSelect resolution (1-{len(RESOLUTIONS)}): ").strip()
            if res_choice in [str(i) for i in range(1, len(RESOLUTIONS) + 1)]:
                resolution = int(RESOLUTIONS[int(res_choice) - 1])
                break
        except ValueError:
            pass

    for key in sorted(PATTERNS.keys(), key=int):
        print(f"  {key}. {PATTERNS[key]['description']}")

    while True:
        pattern_choice = input("\nSelect pattern: ").strip()
        if pattern_choice in PATTERNS:
            pattern_name = PATTERNS[pattern_choice]['name']
            pattern_func = PATTERNS[pattern_choice]['func']
            break

    return resolution, pattern_name, pattern_func

def create_frames(pattern_func, size):
    frame1 = pattern_func(size)

    shift_i = size * 2 // 3
    shift_j = size * 2 // 3

    frame2 = []

    for i in range(size):
        for j in range(size):
            src_i = (i - shift_i) % size
            src_j = (j - shift_j) % size
            frame2.append(frame1[src_i * size + src_j])

    return frame1, frame2


def write_pgm(filename, width, height, pixels):
    with open(filename, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
        f.write(bytes(pixels))


def main():
    resolution, pattern_name, pattern_func = get_user_input()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"{pattern_name}_{resolution}x{resolution}")
    os.makedirs(output_dir, exist_ok=True)

    frame1, frame2 = create_frames(pattern_func, resolution)

    write_pgm(os.path.join(output_dir, "frame_0001.pgm"), resolution, resolution, frame1)
    write_pgm(os.path.join(output_dir, "frame_0002.pgm"), resolution, resolution, frame2)


if __name__ == "__main__":
    main()