import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONSTANTS ---
GAP_LINE = 5
GAP_WORD = 5
GAP_SYMBOL = 1
CANVAS_SIZE = 28
SYMBOL_MAX_DIM = 20


def show_projection(img_slice, projection, title="Projection"):
    """
    Displays the image slice and its corresponding projection graph.
    """
    plt.figure(figsize=(8, 4))

    # Plot the image slice
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title(f"Image: {title}")
    plt.axis('off')

    # Plot the projection graph
    plt.subplot(1, 2, 2)
    # If it's a vertical projection, we plot normally.
    # If it's horizontal, we plot vertically to match the image rows.
    plt.plot(projection)
    plt.title(f"{title} Projection Sum")
    plt.fill_between(range(len(projection)), projection, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_segments(projection, min_gap=5, min_size=2, debug_title=None, img_slice=None):
    """
    Finds start/end indices and optionally visualizes the process.
    """
    # Visualization trigger
    if debug_title and img_slice is not None:
        show_projection(img_slice, projection, debug_title)

    content_indices = np.where(projection > 0)[0]
    if len(content_indices) == 0:
        return []

    segments = []
    start = content_indices[0]

    for i in range(len(content_indices) - 1):
        if content_indices[i + 1] - content_indices[i] > min_gap:
            end = content_indices[i]
            if end - start >= min_size:
                segments.append((start, end))
            start = content_indices[i + 1]

    if content_indices[-1] - start >= min_size:
        segments.append((start, content_indices[-1]))
    return segments


def resize_to_canvas(symbol_img, canvas_size=28, target_dim=20):
    """
    Resizes a symbol with safety checks for zero-dimension images.
    """
    h, w = symbol_img.shape

    # SAFETY: If symbol is empty or essentially invisible, return a blank canvas
    if h == 0 or w == 0:
        return np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = target_dim
        new_h = max(1, int(target_dim / aspect_ratio))  # Ensure at least 1px
    else:
        new_h = target_dim
        new_w = max(1, int(target_dim * aspect_ratio))  # Ensure at least 1px

    resized = cv2.resize(symbol_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    x_off = (canvas_size - new_w) // 2
    y_off = (canvas_size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def process_text_image(image_path):
    # Load and Preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # Invert binary image: text becomes > 0 (white), background becomes 0 (black)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 1. Extract Lines
    image_sum = np.sum(bin_img, axis=1)
    line_segments = get_segments(image_sum, min_gap=GAP_LINE, debug_title="Lines", img_slice=bin_img)

    all_symbols = []

    for l_start, l_end in line_segments:
        line_bin = bin_img[l_start:l_end, :]

        # 2. Extract Words in Line
        word_segments = get_segments(np.sum(line_bin, axis=0), min_gap=GAP_WORD)

        for w_start, w_end in word_segments:
            word_bin = line_bin[:, w_start:w_end]

            # 3. Extract Symbols in Word
            symbol_segments = get_segments(np.sum(word_bin, axis=0), min_gap=GAP_SYMBOL)

            for s_start, s_end in symbol_segments:
                # Extract actual symbol from original grayscale or binary
                symbol_raw = word_bin[:, s_start:s_end]

                # Standardize to 28x28
                symbol_final = resize_to_canvas(symbol_raw)
                all_symbols.append(symbol_final)

    return all_symbols


image_source = "testImages/test2.jpg"
symbols = process_text_image(image_source)

# Quick preview of the first 5 symbols
if symbols:
    fig, axes = plt.subplots(1, min(len(symbols), 5), figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(symbols[i], cmap='gray')
        ax.axis('off')
    plt.show()
