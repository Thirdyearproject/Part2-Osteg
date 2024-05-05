import numpy as np
from sklearn.cluster import KMeans
import imageio.v3 as imageio
import os
import numpy as np


class Pixel:
    def __init__(self, red, green, blue, cluster=-1, min_dist=float("inf")):
        self.red = red
        self.green = green
        self.blue = blue
        self.cluster = cluster
        self.min_dist = min_dist


def kmeans_clustering(pixels, k, epochs, total_pixels):
    # Initialising cluster with -1 as it does not belong to any cluster
    # Initialising min_dist with infinity as it will decrease with number of iterations
    for pixel in pixels:
        pixel.cluster = -1
        pixel.min_dist = float("inf")

    # Initialising clusters with random centroids amongst pixels
    centroids_indices = np.random.choice(total_pixels, k, replace=False)
    centroids = [pixels[i] for i in centroids_indices]

    # Calculate distance, update centroids and repeat for given epochs
    for e in range(epochs):
        # Assigning points to clusters by minimizing euclidean distance
        for i in range(total_pixels):
            for j in range(k):
                # Calculate distance of the point to the centroid
                distance = np.sqrt(
                    (pixels[i].red - centroids[j].red) ** 2
                    + (pixels[i].green - centroids[j].green) ** 2
                    + (pixels[i].blue - centroids[j].blue) ** 2
                )
                if distance < pixels[i].min_dist:
                    pixels[i].min_dist = distance
                    pixels[i].cluster = j

        # Computing new centroids (Heart of k-means i.e. updating centroids)
        # Calculating mean of points in a cluster
        num_pixels = [0.0] * k
        sum_red = [0.0] * k
        sum_green = [0.0] * k
        sum_blue = [0.0] * k

        for i in range(total_pixels):
            cluster = pixels[i].cluster
            num_pixels[cluster] += 1
            sum_red[cluster] += pixels[i].red
            sum_green[cluster] += pixels[i].green
            sum_blue[cluster] += pixels[i].blue

            # Reset distance to max
            pixels[i].min_dist = float("inf")

        # Computing the new centroids
        print(f"Epoch - {e + 1}")
        for i in range(k):
            if (
                num_pixels[i] == 0
            ):  # If no point is assigned to the cluster, assigning new random centroid
                centroids[i] = pixels[np.random.randint(total_pixels)]
            else:  # Else, calculate mean of the points in the cluster and assign them as new centroid
                centroids[i].red = sum_red[i] / num_pixels[i]
                centroids[i].green = sum_green[i] / num_pixels[i]
                centroids[i].blue = sum_blue[i] / num_pixels[i]
            print(
                f"Centroid - {i}: ({centroids[i].red}, {centroids[i].green}, {centroids[i].blue})"
            )
        print()

    # Storing Results
    cluster_count = [0] * k
    for pixel in pixels:
        cluster_count[pixel.cluster] += 1

    # Return cluster counts and pixels
    return cluster_count, pixels


def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Image to pixels
    imagename = input("Enter name of image: ")
    image_path = os.path.join(current_dir, imagename)

    k = int(input("Enter number of clusters(k): "))
    epochs = int(input("Enter number of iterations(epochs): "))

    # Load image
    img = imageio.imread(image_path)
    height, width, _ = img.shape
    total_pixels = height * width

    # Storing pixels
    pixels = [Pixel(*img[i, j]) for i in range(height) for j in range(width)]

    # Training the model
    cluster_count, pixels = kmeans_clustering(pixels, k, epochs, total_pixels)

    # Results
    print(f"Total pixels - {total_pixels}")
    print("Pixels in each cluster:-")
    for c, count in enumerate(cluster_count):
        print(f"Cluster-{c}: {count}")

    # Write pixel information to a text file
    with open(os.path.join(current_dir, "clusters.txt"), "w") as fp:
        for i, pixel in enumerate(pixels):
            fp.write(
                f"{i + 1} {pixel.red:.0f} {pixel.green:.0f} {pixel.blue:.0f} {pixel.cluster}\n"
            )

    # Pixels to Image
    for c in range(k):
        cluster_pixels = [
            pixels[i] for i in range(total_pixels) if pixels[i].cluster == c
        ]
        cluster_length = len(cluster_pixels)
        dim = int(np.ceil(np.sqrt(cluster_length)))  # Square root rounded up
        cluster_image = np.array(
            [[pixel.red, pixel.green, pixel.blue] for pixel in cluster_pixels],
            dtype=np.uint8,
        )
        cluster_image = np.pad(
            cluster_image, ((0, dim**2 - cluster_length), (0, 0)), mode="constant"
        )
        cluster_image = cluster_image.reshape(dim, dim, 3)
        imageio.imwrite(os.path.join(current_dir, f"cluster-{c}.png"), cluster_image)


if __name__ == "__main__":
    main()
