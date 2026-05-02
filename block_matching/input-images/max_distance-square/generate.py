WIDTH = 256
HEIGHT = 256
SQUARE_SIZE = 64

def create_frame(file_name, x_start, y_start):
    with open(file_name, 'wb') as f:
        f.write(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode('ascii'))
        for y in range(HEIGHT):
            for x in range(WIDTH):
                in_square = (x_start <= x < x_start + SQUARE_SIZE and
                             y_start <= y < y_start + SQUARE_SIZE)
                pixel = 200 if in_square else 0
                f.write(bytes([pixel]))

# Frame 1: square in top-left corner
create_frame('test_01.pgm', x_start=0, y_start=0)

# Frame 2: square in bottom-right corner
create_frame('test_02.pgm', x_start=WIDTH-SQUARE_SIZE, y_start=HEIGHT-SQUARE_SIZE)

print("Done.")