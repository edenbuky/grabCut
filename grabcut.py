import numpy as np
import cv2
import argparse
from math import log, pi, sqrt
from igraph import Graph

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
    print("initalize_GMMs V")
    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        print("update_GMMs V")
        #print("BG mean:", bgGMM[0][0], "FG mean:", fgGMM[0][0])
        #print("BG cov:", bgGMM[0][1], "FG cov:", fgGMM[0][1])
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        print("calculate_mincut V")
        print(energy)
        mask = update_mask(mincut_sets, mask)
        print("update_mask V")
        print("Unique mask values:", np.unique(mask))
        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    # Extract background and foreground pixels
    bg_pixels = img[mask == GC_BGD]
    bg_pixels = np.concatenate((bg_pixels, img[mask == GC_PR_BGD]), axis=0)
    fg_pixels = img[mask == GC_PR_FGD]

    # If no fg pixels found (degenerate case), pick something minimal
    if fg_pixels.size == 0:
        fg_pixels = img[mask == GC_FGD]
        if fg_pixels.size == 0:
            fg_pixels = img[0:1, 0:1]  # Just a fallback

    # Compute mean and covariance for bg
    bg_mean = np.mean(bg_pixels, axis=0) if bg_pixels.size > 0 else np.array([0, 0, 0])
    bg_cov = np.cov(bg_pixels.T) if bg_pixels.shape[0] > 1 else np.eye(3)

    # Compute mean and covariance for fg
    fg_mean = np.mean(fg_pixels, axis=0)
    fg_cov = np.cov(fg_pixels.T) if fg_pixels.shape[0] > 1 else np.eye(3)

    bgGMM = [(bg_mean, bg_cov)]
    fgGMM = [(fg_mean, fg_cov)]

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # Re-estimate GMM parameters based on current segmentation
    bg_pixels = img[(mask == GC_BGD) | (mask == GC_PR_BGD)]
    fg_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)]

    if bg_pixels.size > 0:
        bg_mean = np.mean(bg_pixels, axis=0)
        bg_cov = np.cov(bg_pixels.T) if bg_pixels.shape[0] > 1 else np.eye(3)
        bgGMM = [(bg_mean, bg_cov)]
    if fg_pixels.size > 0:
        fg_mean = np.mean(fg_pixels, axis=0)
        fg_cov = np.cov(fg_pixels.T) if fg_pixels.shape[0] > 1 else np.eye(3)
        fgGMM = [(fg_mean, fg_cov)]

    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    # Build graph for mincut
    rows, cols = img.shape[:2]
    N = rows * cols
    src = N
    sink = N + 1

    g = Graph()
    g.add_vertices(N+2)  # all pixels + source + sink

    # Compute unary costs using GMM
    bg_mean, bg_cov = bgGMM[0]
    fg_mean, fg_cov = fgGMM[0]

    bg_inv_cov = np.linalg.inv(bg_cov)
    fg_inv_cov = np.linalg.inv(fg_cov)
    bg_det = np.linalg.det(bg_cov)
    fg_det = np.linalg.det(fg_cov)

    def gmm_prob(x, mean, inv_cov, det):
        diff = (x - mean)
        val = -0.5 * diff @ inv_cov @ diff.T
        return np.exp(val) / ((2 * pi)**(1.5) * sqrt(det))

    edges = []
    capacities = []

    # 4-neighborhood
    directions = [(1,0), (0,1)]

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            pixel = img[r,c,:].astype(np.float64)

            p_bg = gmm_prob(pixel, bg_mean, bg_inv_cov, bg_det)
            p_fg = gmm_prob(pixel, fg_mean, fg_inv_cov, fg_det)

            if p_bg < 1e-15: p_bg = 1e-15
            if p_fg < 1e-15: p_fg = 1e-15

            from_source = -log(p_fg) # cost if we cut towards background is large if fg prob high
            to_sink = -log(p_bg)

            # Connect source->pixel and pixel->sink
            edges.append((src, idx))
            capacities.append(from_source)
            edges.append((idx, sink))
            capacities.append(to_sink)

            # Neighborhood edges
            for d in directions:
                rr = r + d[0]
                cc = c + d[1]
                if 0 <= rr < rows and 0 <= cc < cols:
                    n_idx = rr * cols + cc
                    # small penalty for boundary smoothing
                    w = 1.0
                    edges.append((idx, n_idx))
                    capacities.append(w)
                    edges.append((n_idx, idx))
                    capacities.append(w)

    g.add_edges(edges)
    g.es["capacity"] = capacities

    # Compute mincut
    cut = g.st_mincut(src, sink, capacity="capacity")
    min_cut = [cut.partition[0], cut.partition[1]]
    energy = cut.value

    return min_cut, energy

def update_mask(mincut_sets, mask):
    rows, cols = mask.shape
    N = rows * cols

    if N in mincut_sets[0]:
        source_set = mincut_sets[0]
    else:
        source_set = mincut_sets[1]

    source_pixels = [idx for idx in source_set if idx < N]

    in_source = np.zeros(N, dtype=bool)
    in_source[source_pixels] = True
    in_source = in_source.reshape(rows, cols)

    # הקצאה "קשה" (hard) במקום soft
    mask[in_source] = GC_FGD
    mask[~in_source] = GC_BGD

    return mask

def check_convergence(energy):
    if not hasattr(check_convergence, "prev_energy"):
        check_convergence.prev_energy = None
    if check_convergence.prev_energy is None:
        check_convergence.prev_energy = energy
        return False
    converged = abs(check_convergence.prev_energy - energy) < 1e-3
    check_convergence.prev_energy = energy
    return converged

def cal_metric(predicted_mask, gt_mask):
    pred = predicted_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)

    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    acc = (np.sum(pred == gt)/len(gt)*100) if len(gt)>0 else 0
    jac = (intersection/union*100) if union>0 else 0
    return acc, jac

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
