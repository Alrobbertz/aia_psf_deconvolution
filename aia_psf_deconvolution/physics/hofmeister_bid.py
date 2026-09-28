"""Basic Iterative Deconvolution (BID) and Richardson-Lucy (RL) PSF deconvolution.

Original algorithm and implementation (c) Stefan Hofmeister, adapted from
``PSF-Tools/deconvolve_image.py`` (MIT License, see ``PSF-Tools/LICENSE``) at
https://github.com/stefanhofmeister/PSF-Tools.

Hofmeister et al. (2024), The Basic Iterative Deconvolution: A Fast Instrumental
Point-Spread Function Deconvolution Method That Corrects for Light That Is
Scattered Out of the Field of View of a Detector, Solar Physics, Volume 299,
Issue 6, article id. 77.
https://ui.adsabs.harvard.edu/abs/2023arXiv231211784H/abstract

This adaptation restores the original's optional GPU (CuPy) execution path
behind an explicit ``use_gpu`` flag, defaulting to NumPy/CPU. CuPy is an
optional dependency (install with the ``gpu`` extra); requesting
``use_gpu=True`` without CuPy installed raises a clear error rather than
silently falling back to the CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_cupy() -> Any:
    """Import and return the ``cupy`` module, or raise a clear error."""
    try:
        import cupy
    except ImportError as exc:
        raise ImportError(
            "GPU backend requested (use_gpu=True) but cupy is not installed. "
            "Install the optional extra with `pip install aia_psf_deconvolution[gpu]`, "
            "or call with use_gpu=False to run on CPU via NumPy."
        ) from exc
    return cupy


def deconvolve_bid(
    img: np.ndarray,
    psf: np.ndarray,
    iterations: int = 25,
    tolerance: float = 0.1,
    mask: list[int] | tuple[int, ...] | np.ndarray | None = None,
    use_gpu: bool = False,
    large_psf: bool = False,
    pad: bool = True,
    estimate_background: bool = True,
    constrain_positive: bool = True,
) -> np.ndarray:
    """
    Deconvolve an image with the point spread function

    Perform image deconvolution on an image with the instrument
    point spread function using the bid algorithm published in
    Hofmeister et al. (2024), The Basic Iterative Deconvolution: A Fast Instrumental Point-Spread Function Deconvolution Method That Corrects for Light That Is Scattered Out of the Field of View of a Detector, Solar Physics, Volume 299, Issue 6, article id.77
    https://ui.adsabs.harvard.edu/abs/2023arXiv231211784H/abstract

    Parameters
    ----------
    img : 'numpy 2d array'
        An image.
    psf : `~numpy.ndarray
        The point spread function.
    iterations: `int`
        Maximum number of iterations
    tolerance: 'float'
        The image deconvolution stops when the maximum change from all pixels between the simulated observed image and the observed image is less than TOLERANCE counts.
    mask: 1d array or 2d array
        Allows to select an image subregion for which the convolution is done. By that, the algorithm can massively speeds up. Can be either a 1d array containing the four elements [left, bottom, right, top], or a 2d array masking the pixels that shall be deconvolved. Actually, if the 2d mask is used, from that the boundaries of the 1d array will be calculated, i.e., all pixels in a corresponding rectangualar box will be deconvolved.
        At the moment, the dimensions of mask has to be smaller than half of the dimensions of the image. If you need to deconvolve a larger region, deconvolve the entire image instead.
    use_gpu: True/False
        If True, run the deconvolution on the GPU via CuPy. Requires the optional
        ``cupy`` dependency (``pip install aia_psf_deconvolution[gpu]``); raises
        `ImportError` if requested but unavailable.
    estimate_background: True/False
        If a subregion deconvolution is used, it determines if an incoming scattered light estimate from the surrounding region to the subregion should be applied. This increases the fidelity of the result, but costs some computation time. Generally, it is required for average image intensities and below, but is not required for deconvolving bright image regions.
    pad: True/False
        If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
    large_psf: True/False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and deconvolution is done over the full psf.
    constrain_positive: True/False
        Constrain the deconvolution to positive result intensities. If on, it mitigates small ringing artifacts. If true, allow negative intensities in the reconstructions. Negative intensities are informative, as they can tell that something goes wrong (image calibration artifacts, sligthly inaccurate PSF, ringing, etc.)

    Returns
    -------
    `~numpy.ndarray`
        Deconvolved image

    """
    # At the moment, the mask option only works if the shape of the selected region is smaller than 0.5 * the shape of the image

    # if mask is provided as a list, convert it to a 1d-array
    if isinstance(mask, list) or isinstance(mask, tuple):
        mask = np.array(mask)

    # this factor determines the speed of convergence, and should be set between [0.1, 1.0]
    k = 1.0

    # createa a copy of the image and psf
    img = np.copy(img)
    psf = np.copy(psf)

    # for a psf deconvolution, the length of the axis should be even. Thus, if the mask provieded is odd, make it even by adding one row and/or columng
    if isinstance(mask, np.ndarray):
        bd_box_img, bd_box_psf, bd_box_makeeven = get_boundary_boxes(mask, img, psf)
        if estimate_background:
            # derive the scattered light of the surrounding into the boundary box, and correct the image for it.
            background = estimate_scattered_light(
                img, psf, bd_box_img, large_psf=large_psf, pad=pad, use_gpu=use_gpu
            )
            img = img - background
        # cut the image and psf. Since they are cut, large_psf looses its meaning and thus is set to zero
        img = img[bd_box_img[0] : bd_box_img[1] + 1, bd_box_img[2] : bd_box_img[3] + 1]
        psf = psf[bd_box_psf[0] : bd_box_psf[1] + 1, bd_box_psf[2] : bd_box_psf[3] + 1]
        large_psf = True  # If we cut the image, we definitively want to include scattered light over the entire subimage. Thus, large_psf is set to True. The psf boundary box has already been cut before accordingly.

    # before the psf deconvolution, we have to pad the image and psf with zeros to break the periodic boundary conditions for the convolution in the fourier domain
    if pad:
        img, psf = pad_img_psf(img, psf, large_psf=large_psf, constant_values=np.nan)
        pad_mask = np.isnan(
            img
        )  # the pad mask is required in the next loop - at each iteration, the padding has to be restored.
        img[pad_mask] = 0.0
        psf[np.isnan(psf)] = 0.0
    else:
        pad_mask = np.zeros_like(img, dtype=bool)

    # if we have a gpu, put the arrays on the gpu
    if use_gpu:
        cupy = _require_cupy()
        img = cupy.array(img)
        psf = cupy.array(psf)
        pad_mask = cupy.array(pad_mask)

    # derive the fourier transform of the psf
    psf = np.roll(np.roll(psf, psf.shape[0] // 2, axis=0), psf.shape[1] // 2, axis=1)
    psf = np.fft.rfft2(psf)

    img_decon = np.copy(img)
    for n_iter in range(iterations):
        img_decon_last = np.copy(img_decon)
        # derive the foureir transform of the approximated deconvolved image
        img_decon_con = np.fft.rfft2(img_decon)
        # convolve it with the psf
        img_decon_con = img_decon_con * psf
        # and transform it back to the spatial domain
        img_decon_con = np.fft.irfft2(img_decon_con)
        img_decon_con[pad_mask] = 0.0

        # derive how far we are off between the deconvolved imaged convolved with the psf and the observed image, i.e., how consistent we are
        deviations = img_decon_con - img
        # and adjust the approximated deconvolved image accordingly
        img_decon -= k * deviations
        if constrain_positive:
            img_decon[img_decon < 0] = 0
        # if the deconvolved image has converged, end the iterations
        dev = np.max(np.abs(img_decon - img_decon_last))
        if dev <= tolerance:
            break

    # if we are on the gpu, go back to the cpu
    if use_gpu:
        img_decon = cupy.asnumpy(img_decon)

    # undo the padding
    if pad:
        img_decon, psf = unpad_img_psf(img_decon, psf, large_psf=large_psf)
    # if we had to enlarge the FOV of the mask to get an even pixel length of the axis, shrink the image again
    if isinstance(mask, list) or isinstance(mask, tuple):
        img_decon = img_decon[
            0 - bd_box_makeeven[0] : img_decon.shape[0] - bd_box_makeeven[1] + 1,
            0 - bd_box_makeeven[2] : img_decon.shape[1] - bd_box_makeeven[3] + 1,
        ]

    # and we are done
    img_decon = img_decon.astype(img.dtype)
    return img_decon


def deconvolve_richardson_lucy(
    img: np.ndarray,
    psf: np.ndarray,
    iterations: int = 25,
    pad: bool = True,
    large_psf: bool = False,
    psf_min: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Deconvolve an image with the point spread function

    Perform image deconvolution on an image with the instrument
    point spread function using the Richardson-Lucy deconvolution
    algorithm

    Parameters
    ----------
    img : 'numpy 2d array'
        An image.
    psf : `~numpy.ndarray`
        The point spread function.
    iterations: `int`
        Number of iterations in the Richardson-Lucy algorithm
    pad: True/False
        If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
    large_psf: True/False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and deconvolution is done over the full psf.
    use_gpu: True/False
        If True, run the deconvolution on the GPU via CuPy. Requires the optional
        ``cupy`` dependency (``pip install aia_psf_deconvolution[gpu]``); raises
        `ImportError` if requested but unavailable.

    Returns
    -------
    `~numpy.ndarray`
        Deconvolved image

    Comments:
        Based on the aiapy.deconvolve method, as described in Cheung, M., 2015, *GPU Technology Conference Silicon Valley*, `GPU-Accelerated Image Processing for NASA's Solar Dynamics Observatory <https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s5209-gpu-accelerated+imaging+processing+for+nasa%27s+solar+dynamics+observatory>`_
    """
    img, psf = np.copy(img), np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25 * im_size), int(0.5 * im_size)

    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size + 2 * padsize_large_psf

    # padding is only required if the PSF is not large_psf. Else, the padding of the image has already be done above in the large_psf block.
    if pad and not large_psf:
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size + 2 * padsize_pad
        psf_size = psf_size + 2 * padsize_pad

    # if we have a gpu, put the arrays on the gpu
    if use_gpu:
        cupy = _require_cupy()
        img = cupy.array(img)
        psf = cupy.array(psf)

    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0] // 2, axis=0), psf.shape[1] // 2, axis=1)

    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    psf_conj = psf.conj()

    img_decon = np.copy(img)
    for _ in range(iterations):
        ratio = img / np.fft.irfft2(np.fft.rfft2(img_decon) * psf)
        img_decon = img_decon * np.fft.irfft2(np.fft.rfft2(ratio) * psf_conj)

    # if we are on the gpu, go back to the cpu
    if use_gpu:
        img_decon = cupy.asnumpy(img_decon)

    if large_psf:
        img_decon = img_decon[
            padsize_large_psf : im_size - padsize_large_psf,
            padsize_large_psf : im_size - padsize_large_psf,
        ]

    if pad and not large_psf:
        img_decon = img_decon[
            padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad
        ]

    img_decon = img_decon.astype(img.dtype)

    return img_decon


def convolve_image(
    img: np.ndarray,
    psf: np.ndarray,
    pad: bool = True,
    large_psf: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    """Convolve an image with a point spread function.

    Parameters
    ----------
    img : `~numpy.ndarray`
        An image.
    psf : `~numpy.ndarray`
        The point spread function.
    pad: True/False
        If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
    large_psf: True/False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the convolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and convolution is done over the full psf.
    use_gpu: True/False
        If True, run the convolution on the GPU via CuPy. Requires the optional
        ``cupy`` dependency (``pip install aia_psf_deconvolution[gpu]``); raises
        `ImportError` if requested but unavailable.

    Returns
    -------
    `~numpy.ndarray`
        Convolved image
    """
    img = np.copy(img)
    psf = np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25 * im_size), int(0.5 * im_size)

    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size + 2 * padsize_large_psf

    if pad and not large_psf:
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size + 2 * padsize_pad
        psf_size = psf_size + 2 * padsize_pad

    # if we have a gpu, put the arrays on the gpu
    if use_gpu:
        cupy = _require_cupy()
        img = cupy.array(img)
        psf = cupy.array(psf)

    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0] // 2, axis=0), psf.shape[1] // 2, axis=1)
    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    img_con = np.fft.rfft2(img)
    img_con = img_con * psf
    img_con = np.fft.irfft2(img_con)

    # if we are on the gpu, go back to the cpu
    if use_gpu:
        img_con = cupy.asnumpy(img_con)

    if large_psf:
        img_con = img_con[
            padsize_large_psf : im_size - padsize_large_psf,
            padsize_large_psf : im_size - padsize_large_psf,
        ]

    if pad and not large_psf:
        img_con = img_con[
            padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad
        ]

    img_con = img_con.astype(img.dtype)
    return img_con


def pad_img_psf(
    img: np.ndarray,
    psf: np.ndarray,
    large_psf: bool = False,
    constant_values: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    im_size = np.array(img.shape)
    padsize_pad, padsize_large_psf = (
        (0.25 * im_size).astype(int),
        (0.5 * im_size).astype(int),
    )
    if large_psf:
        img = np.pad(
            img,
            (
                (padsize_large_psf[0], padsize_large_psf[0]),
                (padsize_large_psf[1], padsize_large_psf[1]),
            ),
            constant_values=constant_values,
        )
        img[img == 0] = 0.0  # np.finfo(img.dtype).tiny
    else:
        psf, img = (
            np.pad(
                psf,
                ((padsize_pad[0], padsize_pad[0]), (padsize_pad[1], padsize_pad[1])),
                constant_values=constant_values,
            ),
            np.pad(
                img,
                ((padsize_pad[0], padsize_pad[0]), (padsize_pad[1], padsize_pad[1])),
                constant_values=constant_values,
            ),
        )

    return img, psf


def unpad_img_psf(
    img: np.ndarray, psf: np.ndarray, large_psf: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    im_size = np.array(img.shape)
    unpadsize_pad, unpadsize_large_psf = (
        (1 / 6.0 * im_size).astype(int),
        (1 / 4.0 * im_size).astype(int),
    )
    if large_psf:
        img = img[
            unpadsize_large_psf[0] : im_size[0] - unpadsize_large_psf[0],
            unpadsize_large_psf[1] : im_size[1] - unpadsize_large_psf[1],
        ]
    else:
        img = img[
            unpadsize_pad[0] : im_size[0] - unpadsize_pad[0],
            unpadsize_pad[1] : im_size[1] - unpadsize_pad[1],
        ]
        psf = psf[
            unpadsize_pad[0] : im_size[0] - unpadsize_pad[0],
            unpadsize_pad[1] : im_size[1] - unpadsize_pad[1],
        ]
    return img, psf


def get_boundary_boxes(
    mask: np.ndarray, img: np.ndarray, psf: np.ndarray
) -> tuple[list[int] | np.ndarray, list[int], np.ndarray]:
    if mask.ndim == 1:
        bd_box_img = mask
    if mask.ndim == 2:
        mask = np.where(mask != 0)
        bd_box_img = [min(mask[0]), max(mask[0]), min(mask[1]), max(mask[1])]
    bd_box_makeeven = np.array([0, 0, 0, 0])
    if (bd_box_img[1] - bd_box_img[0]) % 2 == 0:
        bd_box_makeeven[1] = 1
    if (bd_box_img[3] - bd_box_img[2]) % 2 == 0:
        bd_box_makeeven[3] = 1
    if bd_box_img[1] + bd_box_makeeven[1] == img.shape[0]:
        bd_box_makeeven[0] -= 1
        bd_box_makeeven[1] -= 1
    if bd_box_img[3] + bd_box_makeeven[3] == img.shape[1]:
        bd_box_makeeven[2] -= 1
        bd_box_makeeven[3] -= 1
    bd_box_img += bd_box_makeeven

    # convert the boundary box of the mask to a corresponding boundary box for the psf
    bd_box_psf = [
        psf.shape[0] // 2 - (bd_box_img[1] - bd_box_img[0] + 1),
        psf.shape[0] // 2 + (bd_box_img[1] - bd_box_img[0] + 1) - 1,
        psf.shape[1] // 2 - (bd_box_img[3] - bd_box_img[2] + 1),
        psf.shape[1] // 2 + (bd_box_img[3] - bd_box_img[2] + 1) - 1,
    ]
    return bd_box_img, bd_box_psf, bd_box_makeeven


def estimate_scattered_light(
    img_in: np.ndarray,
    psf_in: np.ndarray,
    bd_box_img: list[int] | np.ndarray,
    large_psf: bool = False,
    pad: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    # derive the scattered light into the boundary box region
    img = np.copy(img_in)
    psf = np.copy(psf_in)

    # as we only want to derive the scattered light from the surrounding into the boundary box, set the image intensity in the boundary box to zero.
    # as we only want to have the scattered light, we set the intrinsic intensity, i.e., the center of the psf, to zero.
    img[bd_box_img[0] : bd_box_img[1] + 1, bd_box_img[2] : bd_box_img[3] + 1] = 0.0
    psf[psf.shape[0] // 2, psf.shape[1] // 2] = 0.0

    # pad the image and psf to break the periodic boundary condition involved by the convolution in the fourier domain
    if pad:
        img, psf = pad_img_psf(img, psf, large_psf=large_psf)

    # if we have a gpu, put the arrays on the gpu
    if use_gpu:
        cupy = _require_cupy()
        img = cupy.array(img)
        psf = cupy.array(psf)

    # derive the fourier transform of the psf
    psf = np.roll(np.roll(psf, psf.shape[0] // 2, axis=0), psf.shape[1] // 2, axis=1)
    psf = np.fft.rfft2(psf)
    # derive the foureir transform of the image
    img = np.fft.rfft2(img)
    # convolve it with the psf
    img_background = img * psf
    # and transform it back to the spatial domain
    img_background = np.fft.irfft2(img_background)

    # if we are on the gpu, go back to the cpu
    if use_gpu:
        img_background = cupy.asnumpy(img_background)

    # undo the padding
    if pad:
        img_background, psf = unpad_img_psf(img_background, psf, large_psf=large_psf)

    return img_background
