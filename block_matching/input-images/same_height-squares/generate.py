WIDTH = 256
HEIGHT = 256
SQUARE_SIZE = 64
Y_POS = HEIGHT // 2 - SQUARE_SIZE // 2  # vertically centered

def create_frame(file_name, x_start_left, x_start_right):
    with open(file_name, 'wb') as f:
        f.write(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode('ascii'))
        for y in range(HEIGHT):
            for x in range(WIDTH):
                in_left  = (x_start_left  <= x < x_start_left  + SQUARE_SIZE and Y_POS <= y < Y_POS + SQUARE_SIZE)
                in_right = (x_start_right <= x < x_start_right + SQUARE_SIZE and Y_POS <= y < Y_POS + SQUARE_SIZE)
                pixel = 200 if (in_left or in_right) else 0
                f.write(bytes([pixel]))

# Frame 1: both squares at max horizontal distance
create_frame('test_01.pgm',
             x_start_left=0,
             x_start_right=WIDTH - SQUARE_SIZE)

# Frame 2: both squares shifted right by 32 pixels
create_frame('test_02.pgm',
             x_start_left=32,
             x_start_right=WIDTH - SQUARE_SIZE + 32)

print("Done.")