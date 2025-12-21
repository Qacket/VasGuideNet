import numpy as np
import math
import torch
import torch.nn.functional as F


def limit_values(I, min_val=0.0, max_val=1.0):
    """限制数组元素在指定范围内"""
    return torch.clamp(I, min_val, max_val)


def Hessian3D(I, Sigma):
    if Sigma < 1:
        print("error: Sigma < 1")
        return -1

    I = (
        torch.tensor(I, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )  # Add batch and channel dimensions
    S_round = np.round(3 * Sigma)

    [X, Y, Z] = np.mgrid[
        -S_round : S_round + 1, -S_round : S_round + 1, -S_round : S_round + 1
    ]

    # Construct convolution kernels: Gaussian second derivatives
    DGaussxxx = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (X**2 / pow(Sigma, 2) - 2)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGaussxxy = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (X * Y)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGaussxxz = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (X * Z)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGaussyxx = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (Y * X)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGaussyyy = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (Y**2 / pow(Sigma, 2) - 2)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGaussyyz = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (Y * Z)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGausszzx = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (Z * X)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGausszzy = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (Z * Y)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )
    DGausszzz = (
        (1 / (2 * math.pi * pow(Sigma, 6)))
        * (Z**2 / pow(Sigma, 2) - 2)
        * np.exp(-(X**2 + Y**2 + Z**2) / (2 * pow(Sigma, 2)))
    )

    # Convert kernels to tensors and add batch and channel dimensions
    DGaussxxx = torch.tensor(DGaussxxx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGaussxxy = torch.tensor(DGaussxxy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGaussxxz = torch.tensor(DGaussxxz, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGaussyxx = torch.tensor(DGaussyxx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGaussyyy = torch.tensor(DGaussyyy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGaussyyz = torch.tensor(DGaussyyz, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGausszzx = torch.tensor(DGausszzx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGausszzy = torch.tensor(DGausszzy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    DGausszzz = torch.tensor(DGausszzz, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Perform 3D convolutions
    Dxx = F.conv3d(I, DGaussxxx, padding=S_round.astype(int), stride=1)
    Dyy = F.conv3d(I, DGaussyyy, padding=S_round.astype(int), stride=1)
    Dzz = F.conv3d(I, DGausszzz, padding=S_round.astype(int), stride=1)
    Dxy = F.conv3d(I, DGaussxxy, padding=S_round.astype(int), stride=1)
    Dxz = F.conv3d(I, DGaussxxz, padding=S_round.astype(int), stride=1)
    Dyz = F.conv3d(I, DGaussyyz, padding=S_round.astype(int), stride=1)

    return (
        Dxx.squeeze(0).squeeze(0),
        Dxy.squeeze(0).squeeze(0),
        Dxz.squeeze(0).squeeze(0),
        Dyy.squeeze(0).squeeze(0),
        Dyz.squeeze(0).squeeze(0),
        Dzz.squeeze(0).squeeze(0),
    )


def eig3image(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    hessian_matrix = torch.stack([[Dxx, Dxy, Dxz], [Dxy, Dyy, Dyz], [Dxz, Dyz, Dzz]])

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(hessian_matrix)

    return eigenvalues, eigenvectors


def FrangiFilter3D(I):
    I = limit_values(I)  # 限制值范围
    # I = torch.tensor(I, dtype=torch.float32)  # Convert to tensor
    defaultoptions = {
        "FrangiScaleRange": (1, 10),
        "FrangiScaleRatio": 2,
        "FrangiBetaOne": 0.5,
        "FrangiBetaTwo": 15,
        "verbose": True,
        "BlackWhite": True,
    }
    options = defaultoptions

    sigmas = np.arange(
        options["FrangiScaleRange"][0],
        options["FrangiScaleRange"][1],
        options["FrangiScaleRatio"],
    )
    sigmas.sort()  # Ascending order

    beta = 2 * pow(options["FrangiBetaOne"], 2)
    c = 2 * pow(options["FrangiBetaTwo"], 2)

    shape = (I.shape[0], I.shape[1], I.shape[2], len(sigmas))
    ALLfiltered = np.zeros(shape)

    for i in range(len(sigmas)):
        if options["verbose"]:
            print("Current Frangi Filter Sigma: ", sigmas[i])

        # Make 3D Hessian
        Dxx, Dxy, Dxz, Dyy, Dyz, Dzz = Hessian3D(I, sigmas[i])

        # Correct for scale
        Dxx *= pow(sigmas[i], 2)
        Dyy *= pow(sigmas[i], 2)
        Dzz *= pow(sigmas[i], 2)

        # Calculate (abs sorted) eigenvalues and vectors
        eigenvalues, eigenvectors = eig3image(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)

        Lambda1 = eigenvalues[0]
        Lambda2 = eigenvalues[1]

        Lambda1[Lambda1 == 0] = np.spacing(1)

        Rb = (Lambda2 / Lambda1 + 10e-6) ** 2
        S2 = Lambda1**2 + Lambda2**2

        # Compute the output image
        Ifiltered = np.exp(-Rb / beta) * (np.ones(I.shape) - np.exp(-S2 / c))

        if options["BlackWhite"]:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        # Store the results in 4D matrix
        ALLfiltered[:, :, :, i] = Ifiltered

    # Return the maximum output pixel value for each voxel
    outIm = ALLfiltered.max(3)

    return outIm  # Return as NumPy array


if __name__ == "__main__":
    import cv2

    imagename = "your_pic_name.png"  # Update with your image name
    image = cv2.imread(imagename, cv2.IMREAD_UNCHANGED)
    # Assuming image is 3D, e.g., (depth, height, width)
    blood = cv2.normalize(
        image.astype("double"), None, 0.0, 1.0, cv2.NORM_MINMAX
    )  # Normalize the image
    outIm = FrangiFilter3D(blood)

    img = outIm * 10000  # Scale for display
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
