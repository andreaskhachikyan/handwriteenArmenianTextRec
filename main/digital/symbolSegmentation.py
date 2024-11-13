from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def resize_to_28x28(symbol):
    # Get original dimensions
    original_height, original_width = symbol.shape

    # Set up a 28x28 canvas with white background (255 for binary image)
    canvas = np.ones((28, 28), dtype=np.uint8) * 255

    # Calculate new dimensions to fit the symbol while maintaining aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        # Wider than tall, width is set to 20 and height scaled accordingly
        new_width = 20
        new_height = int(20 / aspect_ratio)
    else:
        # Taller than wide, height is set to 20 and width scaled accordingly
        new_height = 20
        new_width = int(20 * aspect_ratio)

    # Resize the symbol to the new dimensions
    resized_symbol = cv2.resize(symbol, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate padding to center the resized symbol within the 28x28 canvas
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    # Place the resized symbol in the center of the 28x28 canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_symbol

    return canvas


image_name = 'test.png'
image_path = 'testDigitalImage/' + image_name
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

horizontal_projection = np.sum(bin_img, axis=1)

# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.imshow(img)
# plt.subplot(2, 1, 2)
# plt.plot(horizontal_projection)
# plt.title("Sum of Pixels Along Rows (Line Detection)")
# plt.xlabel("Row Index")
# plt.ylabel("Sum of Pixel Values")
# plt.show()

lines_sum = [i for i in range(len(horizontal_projection)) if horizontal_projection[i] > 0]

start = lines_sum[0]
lines = []
for i in range(len(lines_sum) - 1):
    if lines_sum[i + 1] - lines_sum[i] > 5:
        lines.append((start, lines_sum[i]))
        start = lines_sum[i + 1]
lines.append((start, lines_sum[-1]))

lines = [(start - 4, end + 4) for start, end in lines]
print('\n'.join([str(i) for i in lines]))
for start, end in lines:
    line_bin = bin_img[start:end, :]

    # plt.figure(figsize=(10, 6))
    # plt.imshow(line_bin)
    # plt.title("Line")
    # plt.show()

    vertical_sum = np.sum(line_bin, axis=0)

    words_sum = [i for i in range(len(vertical_sum)) if vertical_sum[i] > 0]

    word_start = words_sum[0]
    words = []
    for i in range(len(words_sum) - 1):
        if words_sum[i + 1] - words_sum[i] > 5:
            words.append((word_start, words_sum[i]))
            word_start = words_sum[i + 1]
    words.append((word_start, words_sum[-1]))
    words = [(word_start - 2, word_end + 2) for word_start, word_end in words]
    for word_start, word_end in words:
        word_bin = line_bin[:, word_start:word_end]

        word_vert_sum = np.sum(word_bin, axis=0)
        plt.figure(figsize=(6, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(word_bin)
        plt.subplot(2, 1, 2)
        plt.plot(word_vert_sum)
        plt.show()

        sym_sum = [i for i in range(len(word_vert_sum)) if word_vert_sum[i] > 0]

        sym_start = sym_sum[0]
        symbols = []
        for i in range(len(sym_sum) - 1):
            if sym_sum[i + 1] - sym_sum[i] > 1:
                symbols.append((sym_start, sym_sum[i]))
                sym_start = sym_sum[i + 1]
        symbols.append((sym_start, sym_sum[-1]))

        for sym_start, sym_end in symbols:
            word = img[start:end, word_start:word_end]
            sym_bin = word[:, sym_start:sym_end]

            plt.figure(figsize=(10, 6))
            plt.imshow(sym_bin)
            plt.title("Symbol")
            plt.show()

            sym28x28 = resize_to_28x28(sym_bin)

            plt.figure(figsize=(10, 6))
            plt.imshow(sym28x28)
            plt.title("Symbol")
            plt.show()
