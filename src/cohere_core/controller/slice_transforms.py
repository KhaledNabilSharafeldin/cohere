# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
Generalized slice-by-slice BCDI reconstruction operators.

Implements the forward model psi_j = F_lambda * R * Q_j * rho and the
corresponding gradient for phase retrieval. Handles partial rocking curves
(constant lambda, any J), full rocking curves, and energy scans (variable
lambda) within a single framework.

Operators (in execution order for the forward model):
    Q_j   : Phase gradient — multiplies by exp(i r . Q_j)
    R     : Projection — integrates along kf (sum along axis 2)
    R*    : Backprojection — replicates 2D image along axis 2
    F_lam : Modified 2D FT (plain FFT2D for rocking curves)

Reference: Cha et al., Phys. Rev. Lett. 117, 225501 (2016)
"""

import numpy as np


class SliceTransforms:
    """All slice-by-slice operators for generalized BCDI.

    Parameters
    ----------
    devlib : module
        Device library (nplib, cplib, or torchlib) for array operations.
    slice_data : list of dict
        Per-slice metadata. Each dict contains:
        - 'Q_vector': array(3,) offset from Bragg peak in lab frame
        - 'wavelength': float (constant for rocking curves)
        - 'data': 2D array of measured intensity (already on device)
    geometry : dict
        - 'kf_hat': array(3,) exit beam unit vector
        - 'voxel_size': array(3,) real-space voxel sizes [dx, dy, dz]
    """

    def __init__(self, devlib, slice_data, geometry):
        self.devlib = devlib
        self.J = len(slice_data)
        self.slices = slice_data
        self.geom = geometry

        # Detect scan type from wavelength variance
        wavelengths = [s['wavelength'] for s in slice_data]
        wl_arr = np.array(wavelengths)
        self.lambda_varies = (
            wl_arr.max() - wl_arr.min() > 1e-6 * wl_arr.max()
        ) if wl_arr.max() > 0 else False

        if self.lambda_varies:
            self._init_energy_scan(wavelengths)

        # Build kf-aligned coordinate frame for projection
        self._build_frame()

        # Pre-compute per-slice Q components in the kf frame
        self._precompute_Q_components()

    def _init_energy_scan(self, wavelengths):
        """Set up padding sizes for variable-lambda scans."""
        D = self.geom['det_distance']
        p_det = self.geom['det_pixel']
        lams = np.array(wavelengths)
        delta_lam = np.mean(np.abs(np.diff(lams)))
        self.p_samp = delta_lam * D / p_det
        self.N_pix = [int(l * D / (self.p_samp * p_det)) for l in lams]
        self.N_pix_max = max(self.N_pix)
        self.N_det = self.geom.get('N_det', self.N_pix_max)

    def _build_frame(self):
        """Build kf-aligned coordinate frame (R, R* axes).

        ez = kf_hat (projection axis)
        ex = detector horizontal (orthogonal to ez)
        ey = ez x ex
        """
        kf = np.array(self.geom['kf_hat'], dtype=np.float64)
        ez = kf / np.linalg.norm(kf)

        # Choose a reference direction not parallel to ez
        d1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(d1, ez)) > 0.9:
            d1 = np.array([0.0, 1.0, 0.0])
        ex = d1 - np.dot(d1, ez) * ez
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(ez, ex)

        # frame[i] gives the i-th basis vector
        self.frame = np.stack([ex, ey, ez])  # (3, 3)

    def _precompute_Q_components(self):
        """Transform Q vectors into the kf-aligned frame and store."""
        self.Q_components = []  # list of (Qx, Qy, Qz) in kf frame
        for s in self.slices:
            Q_lab = np.array(s['Q_vector'], dtype=np.float64)
            Q_kf = self.frame @ Q_lab  # project into kf frame
            self.Q_components.append(Q_kf)

    # --- Phase grid construction ---------------------------------

    def _make_phase_grid(self, Q_kf, shape):
        """Build 3D phase array: Q_x*X + Q_y*Y + Q_z*Z.

        Uses real-space coordinates with the voxel sizes from geometry.
        Coordinates are centered (zero at array center).
        """
        dl = self.devlib
        voxel = self.geom.get('voxel_size', [1.0, 1.0, 1.0])

        # Build 1D coordinate arrays centered at the middle
        coords = []
        for dim_i, (n, v) in enumerate(zip(shape, voxel)):
            c = np.arange(n, dtype=np.float64) - n / 2.0
            c = c * v
            coords.append(dl.from_numpy(c.astype(np.float32)))

        # Compute phase = Qx*X + Qy*Y + Qz*Z via broadcasting
        # coords[0] has shape (Nx,), coords[1] (Ny,), coords[2] (Nz,)
        phase = (Q_kf[0] * dl.reshape(coords[0], (shape[0], 1, 1))
                 + Q_kf[1] * dl.reshape(coords[1], (1, shape[1], 1))
                 + Q_kf[2] * dl.reshape(coords[2], (1, 1, shape[2])))
        return phase

    # --- Core operators ------------------------------------------

    def project(self, vol):
        """R: 3D -> 2D, sum along axis 2 (kf direction)."""
        return self.devlib.sum(vol, axis=2)

    def backproject(self, img2d, Nz):
        """R*: 2D -> 3D, replicate along axis 2.

        Creates a 3D volume where every z-slice is a copy of img2d.
        """
        dl = self.devlib
        # expand_dims adds axis=2, then tile repeats along it
        expanded = dl.expand_dims(img2d, axis=2)
        return dl.tile(expanded, (1, 1, Nz))

    def apply_Q(self, rho, j, conjugate=False):
        """Multiply volume by phase factor Q_j = exp(i r . Q_j).

        When conjugate=True, applies Q_j* = exp(-i r . Q_j).
        """
        dl = self.devlib
        Q_kf = self.Q_components[j]
        shape = dl.dims(rho)
        phase = self._make_phase_grid(Q_kf, shape)
        if conjugate:
            phase = -1.0 * phase
        return rho * dl.exp(1j * phase)

    def F_lambda(self, proj_2d, j):
        """Forward modified 2D FT for slice j.

        For constant lambda (rocking curve): plain FFT2D.
        For variable lambda (energy scan): pad -> FFT2D -> crop.
        """
        dl = self.devlib
        if not self.lambda_varies:
            return dl.fftshift(dl.fft(dl.ifftshift(proj_2d), norm='forward'))
        else:
            N = self.N_pix[j]
            padded = self._pad_centered(proj_2d, N)
            ft = dl.fftshift(dl.fft(dl.ifftshift(padded), norm='forward'))
            return self._crop_centered(ft, self.N_det)

    def F_lambda_inv(self, data_2d, j):
        """Inverse modified 2D FT for slice j.

        For constant lambda (rocking curve): plain IFFT2D.
        For variable lambda (energy scan): pad -> IFFT2D -> crop.
        """
        dl = self.devlib
        if not self.lambda_varies:
            # return dl.fftshift(dl.ifft(dl.ifftshift(data_2d), norm='forward'))
            return dl.ifft(data_2d)
        else:
            N = self.N_pix[j]
            padded = self._pad_centered(data_2d, N)
            # ift = dl.fftshift(dl.ifft(dl.ifftshift(padded), norm='forward'))
            ift = dl.ifft(padded)
            return self._crop_centered(ift, self.N_pix_max)

    # --- Main gradient computation -------------------------------

    def compute_gradient(self, rho):
        """Compute the slice-by-slice modulus gradient.

        gradient = sum_j Q_j* R* F_lambda_inv( psi_j - sqrt(I_j) * psi_j / |psi_j| )

        Parameters
        ----------
        rho : 3D complex array
            Current direct-space estimate.

        Returns
        -------
        gradient : 3D complex array, same shape as rho
        error : float, normalized sum-squared modulus error
        """
        dl = self.devlib
        shape = dl.dims(rho)
        Nz = shape[2]
        gradient = dl.zeros(shape).astype(rho.dtype)
        error = 0.0
        norm_factor = 0.0

        for j in range(self.J):
            # Forward: rho -> psi_j
            rho_Q = self.apply_Q(rho, j)
            proj = self.project(rho_Q)
            psi_j = self.F_lambda(proj, j)

            # Modulus residual
            amp = dl.absolute(psi_j)
            amp_safe = dl.where(amp < 1e-30, 1e-30, amp)
            sqrt_I = dl.sqrt(self.slices[j]['data'])
            residual = psi_j - sqrt_I * psi_j / amp_safe

            # Accumulate error
            error += float(dl.sum(dl.square(amp - sqrt_I)))
            norm_factor += float(dl.sum(dl.square(sqrt_I)))

            # Inverse: residual -> volume contribution
            proj_inv = self.F_lambda_inv(residual, j)
            vol = self.backproject(proj_inv, Nz)
            gradient = gradient + self.apply_Q(vol, j, conjugate=True)

        # Normalize gradient by number of slices for proper step size
        if self.J > 0:
            gradient = gradient * (1.0 / self.J)

        # Normalize error
        if norm_factor > 0:
            error = np.sqrt(error / norm_factor)

        return gradient, error

    # --- Padding/cropping helpers (for energy scan) --------------

    def _pad_centered(self, arr, target_size):
        """Pad 2D array symmetrically to target_size x target_size."""
        dl = self.devlib
        shape = dl.dims(arr)
        if shape[0] >= target_size and shape[1] >= target_size:
            return arr
        pad_y = max(0, target_size - shape[0])
        pad_x = max(0, target_size - shape[1])
        top = pad_y // 2
        left = pad_x // 2
        padded = dl.zeros((target_size, target_size)).astype(arr.dtype)
        padded[top:top + shape[0], left:left + shape[1]] = arr
        return padded

    def _crop_centered(self, arr, target_size):
        """Crop 2D array symmetrically to target_size x target_size."""
        dl = self.devlib
        shape = dl.dims(arr)
        if shape[0] <= target_size and shape[1] <= target_size:
            return arr
        crop_y = (shape[0] - target_size) // 2
        crop_x = (shape[1] - target_size) // 2
        return arr[crop_y:crop_y + target_size, crop_x:crop_x + target_size]
