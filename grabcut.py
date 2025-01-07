import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
import igraph as ig
#from PIL import Image

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel


beta = None
global_edges = []
global_weights = []


def grabcut(img, rect, n_iter=5):
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    num_iters = 100
    for i in range(num_iters):
        
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        mask = update_mask(mincut_sets, mask)
        if check_convergence(energy):
            break

    
    mask = finalize_mask(mask)
    return mask, bgGMM, fgGMM

def initalize_GMMs(img, mask, n_components=5):
    global beta
    
    beta = calculate_beta(img)
    initialize_neighbor_edges(img)

    
    bg_pixels = img[mask == GC_BGD]
    fg_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)]

    def initialize_with_kmeans(pixels, n_clusters):
        if len(pixels) == 0:
            return np.zeros((n_clusters, pixels.shape[-1])), np.zeros(n_clusters)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels.reshape(-1, 3))
        centers = kmeans.cluster_centers_
        return centers

    
    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    if len(bg_pixels) > 0:
        centers = initialize_with_kmeans(bg_pixels, n_components)
        bgGMM.means_init = centers
        bgGMM.fit(bg_pixels.reshape(-1, 3))

    
    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    if len(fg_pixels) > 0:
        centers = initialize_with_kmeans(fg_pixels, n_components)
        fgGMM.means_init = centers
        fgGMM.fit(fg_pixels.reshape(-1, 3))

    return bgGMM, fgGMM

def update_GMMs(img, mask, bgGMM, fgGMM):
    
    bg_pixels = img[(mask == GC_BGD) | (mask == GC_PR_BGD)]
    fg_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)]

    
    if len(bg_pixels) > 0:
        bgGMM.fit(bg_pixels.reshape((-1, img.shape[-1])))

    
    if len(fg_pixels) > 0:
        fgGMM.fit(fg_pixels.reshape((-1, img.shape[-1])))

    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    num_pixels = img.shape[0] * img.shape[1]
    source = num_pixels
    sink = num_pixels + 1

    img_flat = img.reshape(-1, img.shape[-1])
    fg_probs = -fgGMM.score_samples(img_flat).reshape(img.shape[:-1])
    bg_probs = -bgGMM.score_samples(img_flat).reshape(img.shape[:-1])

    lam = 2 * max(np.abs(fg_probs.max()), np.abs(bg_probs.max()))
    
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pixels + 2)

    edges = global_edges.copy()
    weights = global_weights.copy()

    def vid(i, j): 
        return (img.shape[1] * i) + j

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            if mask[i, j] == GC_FGD or mask[i, j] ==GC_BGD:
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

    #graph.add_edges(edges, attributes={'weight': weights})
    graph.add_edges(edges)
    graph.es['weight'] = weights
    min_cut = graph.st_mincut(source, sink, capacity="weight")
    bg_segment = min_cut.partition[0]
    fg_segment = min_cut.partition[1]

    combined_segments = [bg_segment, fg_segment]
    energy = min_cut.value

    return combined_segments, energy
















def update_mask(mincut_sets, mask):
    bg_segment, fg_segment = mincut_sets
    num_pixels = mask.shape[0] * mask.shape[1]
    source = num_pixels
    sink = num_pixels + 1

    
    if source in bg_segment:
        bg_segment, fg_segment = fg_segment, bg_segment

    
    mask_flat = mask.flatten()

    
    fg_segment = np.array([v for v in fg_segment if v not in (source, sink)])
    bg_segment = np.array([v for v in bg_segment if v not in (source, sink)])

    
    if fg_segment.size > 0:
        fg_indices = fg_segment[mask_flat[fg_segment] == GC_PR_BGD]
        mask_flat[fg_indices] = GC_PR_FGD

    
    if bg_segment.size > 0:
        bg_indices = bg_segment[mask_flat[bg_segment] == GC_PR_FGD]
        mask_flat[bg_indices] = GC_PR_BGD


    new_mask = mask_flat.reshape(mask.shape)

    return new_mask

def check_convergence(energy):
    if not hasattr(check_convergence, "prev_energy"):
        check_convergence.prev_energy = None
    if check_convergence.prev_energy is None:
        check_convergence.prev_energy = energy
        return False
    converged = abs(check_convergence.prev_energy - energy) < 0.0001 # 1e-4
    #print(abs(check_convergence.prev_energy - energy))
    check_convergence.prev_energy = energy
    return converged

def cal_metric(predicted_mask, gt_mask):

    intersection = np.logical_and(predicted_mask, gt_mask).sum()
    union = np.logical_or(predicted_mask, gt_mask).sum()


    jaccard = intersection / union if union != 0 else 1.0


    accuracy = (predicted_mask == gt_mask).sum() / predicted_mask.size

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

    total_connections = 4 * h * w - 3 * (h + w) + 2  
    beta /= total_connections

 
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


    mask_flat[mask_flat == GC_PR_BGD] = GC_BGD


    mask_flat[mask_flat == GC_PR_FGD] = GC_FGD

    final_mask = mask_flat.reshape(mask.shape)

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
