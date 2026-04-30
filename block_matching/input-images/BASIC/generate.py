WIDTH = 256
HEIGHT = 256

def crea_immagine_pattern(nome_file, x_start, y_start):
    x_end = x_start + 64
    y_end = y_start + 64
    x_mid = x_start + 32
    y_mid = y_start + 32

    with open(nome_file, 'wb') as f:
        f.write(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode('ascii'))
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if x_start <= x < x_end and y_start <= y < y_end:
                    if   x < x_mid and y < y_mid:  pixel = 255
                    elif x >= x_mid and y < y_mid:  pixel = 192
                    elif x < x_mid and y >= y_mid:  pixel = 128
                    else:                            pixel = 64
                else:
                    pixel = 0
                f.write(bytes([pixel]))

# Frame 1: quadrato a (64, 64)
crea_immagine_pattern('test_01.pgm', x_start=64, y_start=64)

# Frame 2: spostato solo a destra di 32px, y INVARIATA
crea_immagine_pattern('test_02.pgm', x_start=96, y_start=64)

print("Fatto!")