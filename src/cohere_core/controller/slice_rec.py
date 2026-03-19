# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
Slice-by-slice BCDI reconstruction engine.

SliceRec extends the existing Rec class to support reconstruction from any
sparse subset of rocking-curve slices through the Bragg peak. The existing
3D FFT pipeline (Rec) is completely untouched. SliceRec overrides only the
reciprocal-space operations, fusing the forward transform, modulus constraint,
and inverse transform into a single gradient computation via SliceTransforms.

Usage:
    Set slice_mode = true in config_rec and use algorithm mnemonics
    slER, slHIO, slSF, or slRAAR in the algorithm_sequence.
"""

import os
import numpy as np
import cohere_core.utilities.utils as ut
from cohere_core.controller.phasing import Rec
from cohere_core.controller.slice_transforms import SliceTransforms
import cohere_core.controller.phasing as _phasing


class SliceRec(Rec):
    """Slice-by-slice BCDI reconstruction.

    Inherits all direct-space operations (er, hio, sf, raar, shrink wrap,
    twin, etc.) unchanged from Rec. Overrides the reciprocal-space cycle
    with a fused slice_modulus operation that computes the modulus gradient
    across all available slices.
    """

    def __init__(self, params, data_file, pkg, **kwargs):
        super().__init__(params, data_file, pkg, **kwargs)

        # Append slice_modulus to iter_functions so op_flow can reference it
        self.iter_functions.append(self.slice_modulus)

    def init_dev(self, device_id):
        """Initialize device, load data, then set up slice transforms."""
        ret = super().init_dev(device_id)
        if ret < 0:
            return ret

        self._init_slice_transforms()
        return 0

    def _init_slice_transforms(self):
        """Build SliceTransforms from loaded data and config params."""
        # The loaded 3D data cube has shape (Nframes, Ny, Nx) after fftshift.
        # Each frame along axis 0 is one rocking-curve slice.
        data_3d = self.data  # already on device, fftshifted

        # Determine which frames to use
        n_frames = self.dims[0]
        slice_indices = self.params.get('slice_indices', list(range(n_frames)))

        # Get geometry from params
        scan_type = self.params.get('scan_type', 'rocking')
        wavelength = self.params.get('wavelength', 1.0)
        kf_hat = self.params.get('kf_direction', [0.0, 0.0, 1.0])
        voxel_size = self.params.get('voxel_size', [1.0, 1.0, 1.0])

        # Build per-slice metadata
        slice_data = []
        if 'Q_vectors' in self.params:
            # Q vectors provided directly
            Q_vectors = self.params['Q_vectors']
        else:
            # Compute Q vectors from scan angles for rocking curve
            Q_vectors = self._compute_Q_from_angles(slice_indices)

        for idx_in_list, frame_idx in enumerate(slice_indices):
            slice_info = {
                'Q_vector': Q_vectors[idx_in_list],
                'wavelength': wavelength,
                'data': data_3d[frame_idx],  # 2D slice of measured intensity
            }
            if scan_type == 'energy' and 'wavelengths' in self.params:
                slice_info['wavelength'] = self.params['wavelengths'][idx_in_list]
            slice_data.append(slice_info)

        geometry = {
            'kf_hat': kf_hat,
            'voxel_size': voxel_size,
        }
        if scan_type == 'energy':
            geometry['det_distance'] = self.params['det_distance']
            geometry['det_pixel'] = self.params['det_pixel']
            geometry['N_det'] = self.params.get('N_det', self.dims[2])

        self.slice_transforms = SliceTransforms(_phasing.devlib, slice_data, geometry)

        # Store slice count for reference
        self.n_slices = len(slice_indices)

    def _compute_Q_from_angles(self, slice_indices):
        """Compute Q_j vectors from rocking curve frame indices.

        For a rocking curve, Q_j traces a path perpendicular to the Bragg
        peak. The z-component of Q in the kf frame is proportional to
        the frame offset from center.
        """
        n_frames = self.dims[0]
        center = n_frames / 2.0

        # Angular step per frame (in the kf-aligned frame, Q is along z)
        # The Q spacing is 2*pi / (N * dz) where dz is the voxel size along z
        voxel_size = self.params.get('voxel_size', [1.0, 1.0, 1.0])
        dq_z = 2.0 * np.pi / (n_frames * voxel_size[2]) if voxel_size[2] > 0 else 0.0

        Q_vectors = []
        for idx in slice_indices:
            # Q_j in lab frame: for standard rocking, offset is along kf
            kf = np.array(self.params.get('kf_direction', [0.0, 0.0, 1.0]),
                          dtype=np.float64)
            kf = kf / np.linalg.norm(kf)
            q_offset = (idx - center) * dq_z
            Q_vectors.append(kf * q_offset)

        return Q_vectors

    #=============Overrides=============================

    def slice_modulus(self):
        """Fused forward-residual-inverse operation for slice-by-slice.

        Computes the modulus gradient across all slices and applies the
        modulus projection: ds_image_proj = rho - 0.5 * gradient.
        """
        gradient, error = self.slice_transforms.compute_gradient(self.ds_image)
        self.errs.append(error)

        # Modulus projection P_m: rho - 0.5 * gradient
        self.ds_image_proj = self.ds_image - 0.5 * gradient

    def to_reciprocal_space(self):
        """No-op: slice_modulus handles the forward transform."""
        pass

    def modulus(self):
        """No-op: slice_modulus handles the modulus constraint."""
        pass

    def to_direct_space(self):
        """No-op: slice_modulus handles the inverse transform."""
        pass
