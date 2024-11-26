import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img1 = cv2.imread(cv2.samples.findFile("leuvenA.jpg"), 0)
img2 = cv2.imread(cv2.samples.findFile("leuvenA.jpg"), 0)
# increase brightness of img2
img2 = cv2.add(img2, 50)
print(img1.shape, img2.shape)

def extract_patches(image, corners, patch_size=16):
    patches = []
    half_size = patch_size // 2
    padded_image = cv2.copyMakeBorder(image, half_size, half_size, half_size, half_size, cv2.BORDER_REFLECT)
    
    for corner in corners:
        x, y = corner.ravel()
        x, y = int(x) + half_size, int(y) + half_size
        patch = padded_image[y-half_size:y+half_size, x-half_size:x+half_size]
        patch = (patch - np.mean(patch)) / np.std(patch)  # Normalize the patch
        patches.append(patch)
        
    return np.array(patches)


def get_corners(image):
    # Detect corners using Harris Corner Detector
    corners = cv2.cornerHarris(image, 2, 3, 0.04)

    # Dilate corner image to enhance corner points
    corners = cv2.dilate(corners, None)

    # Threshold to get corner points
    ret, dst = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # Define criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Display detected corners
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv2.circle(image, (int(x), int(y)), 3, 255, -1)

    # cv2.imshow('Corners', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return corners
corners1 = get_corners(img1)
corners2 = get_corners(img2)

corners1 = np.int0(corners1)
corners2 = np.int0(corners2)

patches1 = extract_patches(img1, corners1)
patches2 = extract_patches(img2, corners2)

def match_descriptors(patches1, patches2):
    matches = []
    for i, patch1 in enumerate(patches1):
        min_dist = float('inf')
        best_match = -1
        for j, patch2 in enumerate(patches2):
            dist = np.linalg.norm(patch1 - patch2)
            if dist < min_dist:
                min_dist = dist
                best_match = j
        if min_dist < 1:
            matches.append((i, best_match))
    return matches

# Draw matches
matches = match_descriptors(patches1, patches2)
matches = np.array(matches).astype(np.int32)
print(corners2)
width = img1.shape[1]
img_matches = cv2.hconcat([img1, img2])
print(matches)
for i, j in matches:
    cv2.line(img_matches, tuple(corners1[i].ravel()), tuple((corners2[j] + [width, 0]).ravel()), 255, 1)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(10)