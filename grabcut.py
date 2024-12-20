import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

# Global variable for beta
beta = None
global_edges = []
global_weights = []
# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    print("initalize_GMMs completed")
    num_iters = 100
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        print("update_GMMs completed")
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        print("calculate_mincut completed")
        mask = update_mask(mincut_sets, mask)
        print("update_mask completed")
        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    mask = finalize_mask(mask)
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    global beta
    beta = calculate_beta(img)
    print(f"Initialized beta: {beta}")
    initialize_neighbor_edges(img)

    # Extract background and foreground pixels based on the mask
    bg_pixels = img[mask == GC_BGD]
    fg_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)]

    print(f"Initializing GMMs: {len(bg_pixels)} background pixels, {len(fg_pixels)} foreground pixels")

    def initialize_with_kmeans(pixels, n_clusters):
        if len(pixels) == 0:
            print("Warning: No pixels for k-means initialization.")
            return np.zeros((n_clusters, pixels.shape[-1])), np.zeros(n_clusters)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels.reshape(-1, 3))
        centers = kmeans.cluster_centers_
        print(f"k-means centers shape: {centers.shape}")
        return centers

    # Initialize GMM for background
    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    if len(bg_pixels) > 0:
        centers = initialize_with_kmeans(bg_pixels, n_components)
        bgGMM.means_init = centers
        bgGMM.fit(bg_pixels.reshape(-1, 3))
        print("Initialized and fitted background GMM.")
    else:
        print("Warning: No background pixels for GMM initialization.")

    # Initialize GMM for foreground
    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    if len(fg_pixels) > 0:
        centers = initialize_with_kmeans(fg_pixels, n_components)
        fgGMM.means_init = centers
        fgGMM.fit(fg_pixels.reshape(-1, 3))
        print("Initialized and fitted foreground GMM.")
    else:
        print("Warning: No foreground pixels for GMM initialization.")

    return bgGMM, fgGMM
def update_GMMs(img, mask, bgGMM, fgGMM):
    # Extract background and foreground pixels based on the mask
    bg_pixels = img[(mask == GC_BGD) | (mask == GC_PR_BGD)]
    fg_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)]

    print(f"Updating GMMs: {len(bg_pixels)} background pixels, {len(fg_pixels)} foreground pixels")

    # Update GMM for background
    if len(bg_pixels) > 0:
        # We don't need to re-initialize with k-means here since we're updating
        bgGMM.fit(bg_pixels.reshape((-1, img.shape[-1])))
        print("Updated background GMM.")
    else:
        print("Warning: No background pixels for GMM update.")

    # Update GMM for foreground
    if len(fg_pixels) > 0:
        # Similarly, no re-initialization needed for update
        fgGMM.fit(fg_pixels.reshape((-1, img.shape[-1])))
        print("Updated foreground GMM.")
    else:
        print("Warning: No foreground pixels for GMM update.")

    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    num_pixels = img.shape[0] * img.shape[1]
    source = num_pixels
    sink = num_pixels + 1
    print(f"calculate_mincut: Starting with {num_pixels} pixels.")

    img_flat = img.reshape(-1, img.shape[-1])
    fg_probs = -fgGMM.score_samples(img_flat).reshape(img.shape[:-1])
    bg_probs = -bgGMM.score_samples(img_flat).reshape(img.shape[:-1])

    print(f"Foreground probabilities: min={fg_probs.min()}, max={fg_probs.max()}")
    print(f"Background probabilities: min={bg_probs.min()}, max={bg_probs.max()}")

    lam = 2 * max(np.abs(fg_probs.max()), np.abs(bg_probs.max()))
    # Create graph with relevant vertices
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pixels + 2)

    edges = global_edges.copy()
    weights = global_weights.copy()

    def vid(i, j): # vertex ID
        return (img.shape[1] * i) + j

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            # add edges to source and sink
            if mask[i, j] == GC_FGD or GC_BGD:
                if mask[i, j] == GC_FGD:
                    edges.append((vid(i, j), source))
                    weights.append(lam)
                else:
                    edges.append((vid(i, j), sink))
                    weights.append(lam)
            else:
                edges.append((vid(i, j), source))
                weights.append(bg_probs[i, j])

                edges.append((vid(i, j), sink))
                weights.append(fg_probs[i, j])

    graph.add_edges(edges, attributes={'weight': weights})

    min_cut = graph.st_mincut(source, sink, capacity="weight")
    bg_segment = min_cut.partition[0]
    fg_segment = min_cut.partition[1]

    print("calculate_mincut: Computed min-cut. Source size: ", len(bg_segment), ", Sink size: ",len(fg_segment))

    # Return segments and energy
    combined_segments = [bg_segment, fg_segment]
    energy = min_cut.value

    print(f"calculate_mincut: Finished with                               ***Energy*** {energy}.")
    return combined_segments, energy
def update_mask(mincut_sets, mask):
    bg_segment, fg_segment = mincut_sets
    num_pixels = mask.shape[0] * mask.shape[1]
    source = num_pixels
    sink = num_pixels + 1

    # Ensure fg_segment is the one with foreground pixels
    if source in bg_segment:
        bg_segment, fg_segment = fg_segment, bg_segment

    # Create a flat version of the mask for easier updates
    mask_flat = mask.flatten()

    # Convert fg_segment and bg_segment to NumPy arrays for vectorized operations
    fg_segment = np.array([v for v in fg_segment if v not in (source, sink)])
    bg_segment = np.array([v for v in bg_segment if v not in (source, sink)])

    # Update probable background pixels in fg_segment to probable foreground
    fg_indices = fg_segment[mask_flat[fg_segment] == GC_PR_BGD]
    mask_flat[fg_indices] = GC_PR_FGD

    # Update probable foreground pixels in bg_segment to probable background
    bg_indices = bg_segment[mask_flat[bg_segment] == GC_PR_FGD]
    mask_flat[bg_indices] = GC_PR_BGD

    # Reshape the mask back to its original dimensions
    new_mask = mask_flat.reshape(mask.shape)

    print(f"update_mask: Updated mask - Probable background: {(new_mask == GC_PR_BGD).sum()}, "
          f"Probable foreground: {(new_mask == GC_PR_FGD).sum()}")
    print(f"update_mask: Hard background unchanged: {(new_mask == GC_BGD).sum()}, "
          f"Hard foreground unchanged: {(new_mask == GC_FGD).sum()}")

    return new_mask


def check_convergence(energy):
    if not hasattr(check_convergence, "prev_energy"):
        check_convergence.prev_energy = None
    if check_convergence.prev_energy is None:
        check_convergence.prev_energy = energy
        print(f"First energy value: {energy}")
        return False
    converged = abs(check_convergence.prev_energy - energy) < 100
    print(f"Energy difference: {abs(check_convergence.prev_energy - energy)}, Converged: {converged}")
    check_convergence.prev_energy = energy
    return converged

def cal_metric(predicted_mask, gt_mask):
    # Calculate intersection and union for Jaccard index (IoU)
    intersection = np.logical_and(predicted_mask, gt_mask).sum()
    union = np.logical_or(predicted_mask, gt_mask).sum()

    # Calculate Jaccard index
    jaccard = intersection / union if union != 0 else 1.0

    # Calculate accuracy
    accuracy = (predicted_mask == gt_mask).sum() / predicted_mask.size

    print(f"Metrics - Accuracy: {accuracy * 100:.2f}%, Jaccard: {jaccard * 100:.2f}%")

    return accuracy * 100, jaccard * 100

def calculate_beta(img):
    h, w, _ = img.shape
    beta = 0

    for i in range(h):
        for j in range(w):
            if i > 0:
                diff = img[i, j] - img[i - 1, j]
                beta += diff.dot(diff)
            if j > 0:
                diff = img[i, j] - img[i, j - 1]
                beta += diff.dot(diff)
            if i > 0 and j > 0:
                diff = img[i, j] - img[i - 1, j - 1]
                beta += diff.dot(diff)
            if i > 0 and j < w - 1:
                diff = img[i, j] - img[i - 1, j + 1]
                beta += diff.dot(diff)
    print(f"Sum of the square of the color differences for all connections of beta : {beta}")
    # Normalize beta based on the total number of connections (including diagonals)
    total_connections = 4 * h * w - 3 * (h + w) + 2  # The formula for total connections with diagonals
    print(f"Total number of connections : {total_connections}")
    beta /= total_connections

    # Adjust beta
    beta *= 2
    beta = 1 / beta

    return beta
def initialize_neighbor_edges(img):
    """ Precompute edges and weights for neighboring pixels """
    global global_edges, global_weights
    h, w, _ = img.shape
    edges = []
    weights = []

    def compute_V(i, j, oi, oj, gamma=50):
        diff = img[i, j] - img[oi, oj]
        return gamma * np.exp(- beta * diff.dot(diff))

    def vid(i, j):
        return (img.shape[1] * i) + j

    for i in range(h):
        for j in range(w):
            # Add edges to neighbors
            if i > 0:
                oi, oj = i - 1, j
                edges.append((vid(i, j), vid(oi, oj)))
                weights.append(compute_V(i, j, oi, oj))
            if j > 0:
                oi, oj = i, j - 1
                edges.append((vid(i, j), vid(oi, oj)))
                weights.append(compute_V(i, j, oi, oj))
            if i > 0 and j > 0:
                oi, oj = i - 1, j - 1
                edges.append((vid(i, j), vid(oi, oj)))
                weights.append(compute_V(i, j, oi, oj))
            if i > 0 and j < w - 1:
                oi, oj = i - 1, j + 1
                edges.append((vid(i, j), vid(oi, oj)))
                weights.append(compute_V(i, j, oi, oj))

    global_edges = edges
    global_weights = weights
def finalize_mask(mask):
    mask_flat = mask.flatten()

    # Convert probable background to hard background
    mask_flat[mask_flat == GC_PR_BGD] = GC_BGD

    # Convert probable foreground to hard foreground
    mask_flat[mask_flat == GC_PR_FGD] = GC_FGD

    # Reshape back to original dimensions
    final_mask = mask_flat.reshape(mask.shape)

    print(f"finalize_mask: Finalized mask - hard background: {(final_mask == GC_BGD).sum()}, "
          f"hard foreground: {(final_mask == GC_FGD).sum()}")

    return final_mask


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        x1, y1, x2, y2 = map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' '))
        rect = (x1, y1, x2 - x1, y2 - y1)
        #rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
