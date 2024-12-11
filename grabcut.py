import numpy as np
import cv2
import argparse

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel


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

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    n_components = 5
    # Reshape image into a 2D array of pixels (N x 3 for RGB)
    pixels = img.reshape(-1, img.shape[2])

    # Extract foreground and background pixels based on the mask
    fg_pixels = pixels[mask.flatten() == GC_PR_FGD]
    bg_pixels = pixels[mask.flatten() == GC_BGD]

    def run_kmeans(data, n_clusters):
        """Helper function to run OpenCV KMeans."""
        data = np.float32(data)  # Convert to float32 for cv2.kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return labels, centers

    # Run KMeans for foreground and background
    fg_labels, fg_centers = run_kmeans(fg_pixels, n_components)
    bg_labels, bg_centers = run_kmeans(bg_pixels, n_components)

    def build_gmm(pixels, labels, centers, n_clusters):
        """Helper function to build GMM manually."""
        gmm = {
            "weights": np.zeros(n_clusters),
            "means": centers,
            "covariances": []
        }
        for i in range(n_clusters):
            cluster_pixels = pixels[labels.flatten() == i]
            gmm["weights"][i] = len(cluster_pixels) / len(pixels)
            if len(cluster_pixels) > 0:
                gmm["covariances"].append(np.cov(cluster_pixels, rowvar=False) + 1e-6 * np.eye(cluster_pixels.shape[1]))
            else:
                gmm["covariances"].append(np.eye(centers.shape[1]))
        return gmm

    # Build GMMs for foreground and background
    fgGMM = build_gmm(fg_pixels, fg_labels, fg_centers, n_components)
    bgGMM = build_gmm(bg_pixels, bg_labels, bg_centers, n_components)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    def assign_pixels_to_components(pixels, gmm):
        """Assign each pixel to the Gaussian component with the highest likelihood."""
        num_components = len(gmm["means"])
        likelihoods = np.zeros((pixels.shape[0], num_components))
        for i in range(num_components):
            mean = gmm["means"][i]
            cov = gmm["covariances"][i]
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            diff = pixels - mean
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            likelihoods[:, i] = gmm["weights"][i] * np.exp(exponent) / np.sqrt((2 * np.pi) ** 3 * det_cov)
        return np.argmax(likelihoods, axis=1)

    def update_gmm_params(pixels, labels, num_components):
        """Update weights, means, and covariances for GMM."""
        weights = np.zeros(num_components)
        means = np.zeros((num_components, pixels.shape[1]))
        covariances = []
        for i in range(num_components):
            cluster_pixels = pixels[labels == i]
            weights[i] = len(cluster_pixels) / len(pixels)
            if len(cluster_pixels) > 0:
                means[i] = np.mean(cluster_pixels, axis=0)
                covariances.append(np.cov(cluster_pixels, rowvar=False) + 1e-6 * np.eye(pixels.shape[1]))
            else:
                covariances.append(np.eye(pixels.shape[1]))
        return {"weights": weights, "means": means, "covariances": covariances}

    # Reshape image into a 2D array of pixels (N x 3 for RGB)
    pixels = img.reshape(-1, img.shape[2])

    # Extract foreground and background pixels based on the mask
    fg_pixels = pixels[mask.flatten() == GC_PR_FGD]
    bg_pixels = pixels[mask.flatten() == GC_BGD]

    # Assign pixels to GMM components
    fg_labels = assign_pixels_to_components(fg_pixels, fgGMM)
    bg_labels = assign_pixels_to_components(bg_pixels, bgGMM)

    # Update GMM parameters
    fgGMM = update_gmm_params(fg_pixels, fg_labels, len(fgGMM["means"]))
    bgGMM = update_gmm_params(bg_pixels, bg_labels, len(bgGMM["means"]))

    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    global previous_energy
    # Reshape image into a 2D array of pixels (N x 3 for RGB)
    h, w, c = img.shape
    pixels = img.reshape(-1, c)
    n_pixels = pixels.shape[0]

    # Create a graph
    graph = ig.Graph()
    graph.add_vertices(n_pixels + 2)  # Pixels + source + sink
    source = n_pixels
    sink = n_pixels + 1

    # Compute beta (for N-links)
    beta = 1 / (2 * np.mean(np.var(pixels, axis=0)))

    def compute_nlink_weight(pixel1, pixel2):
        """Compute the weight of the N-link between two neighboring pixels."""
        diff = np.linalg.norm(pixel1 - pixel2)
        return np.exp(-beta * diff ** 2)

    # Add N-links (neighbors in 4-connectivity)
    for y in range(h):
        for x in range(w):
            pixel_index = y * w + x
            pixel_color = pixels[pixel_index]

            if x + 1 < w:  # Right neighbor
                neighbor_index = pixel_index + 1
                neighbor_color = pixels[neighbor_index]
                weight = compute_nlink_weight(pixel_color, neighbor_color)
                graph.add_edge(pixel_index, neighbor_index, weight=weight)

            if y + 1 < h:  # Bottom neighbor
                neighbor_index = pixel_index + w
                neighbor_color = pixels[neighbor_index]
                weight = compute_nlink_weight(pixel_color, neighbor_color)
                graph.add_edge(pixel_index, neighbor_index, weight=weight)

    def compute_tlink_weight(pixel, gmm, target):
        """Compute the weight of the T-link to the source or sink."""
        total_weight = 0
        for i in range(len(gmm["weights"])):
            mean = gmm["means"][i]
            cov = gmm["covariances"][i]
            weight = gmm["weights"][i]
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            diff = pixel - mean
            exponent = -0.5 * diff @ inv_cov @ diff.T
            gaussian = weight * np.exp(exponent) / np.sqrt((2 * np.pi) ** c * det_cov)
            total_weight += gaussian
        return -np.log(total_weight + 1e-10) if target == "source" else np.log(total_weight + 1e-10)

    # Add T-links (source and sink connections)
    for pixel_index, pixel in enumerate(pixels):
        if mask.flatten()[pixel_index] == GC_PR_FGD:
            fg_weight = compute_tlink_weight(pixel, fgGMM, "source")
            bg_weight = compute_tlink_weight(pixel, bgGMM, "sink")
            graph.add_edge(source, pixel_index, weight=fg_weight)
            graph.add_edge(pixel_index, sink, weight=bg_weight)
        elif mask.flatten()[pixel_index] == GC_FGD:
            graph.add_edge(source, pixel_index, weight=float("inf"))
        elif mask.flatten()[pixel_index] == GC_BGD:
            graph.add_edge(pixel_index, sink, weight=float("inf"))

    # Perform graph cut
    cut = graph.st_mincut(source, sink)

    # Prepare the min_cut result
    min_cut = [[], []]
    for i in range(n_pixels):
        if i in cut.partition[0]:
            min_cut[0].append(i)  # Foreground
        else:
            min_cut[1].append(i)  # Background

    # Energy of the cut
    energy = cut.value

    return min_cut, energy


def update_mask(mincut_sets, mask):
    # Flatten the mask for easier indexing
    flat_mask = mask.flatten()

    # Update foreground pixels
    for pixel_index in mincut_sets[0]:  # Foreground pixels
        if flat_mask[pixel_index] != GC_FGD:  # Preserve hard constraints
            flat_mask[pixel_index] = GC_PR_FGD

    # Update background pixels
    for pixel_index in mincut_sets[1]:  # Background pixels
        if flat_mask[pixel_index] != GC_BGD:  # Preserve hard constraints
            flat_mask[pixel_index] = GC_PR_BGD

    # Reshape the mask back to its original dimensions (if needed)
    mask[:] = flat_mask.reshape(mask.shape)  # Update in-place

    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100

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
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
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
