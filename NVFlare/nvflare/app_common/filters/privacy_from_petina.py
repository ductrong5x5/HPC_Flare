# Filename: privacy_from_petina.py
# Description: Implementation of Exponential, Gaussian, Laplace and Adaptive clipping Privacy Filter for differential privacy in NVFlare.
# Developer: Sharmin Afrose
# Date: 2024-11-19


from typing import List, Union

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from PETINA import DP_Mechanisms


class PrivacyPetinaLibrary(DXOFilter):
    def __init__(self, gamma=0.0001, epsilon=1.0, delta=10e-5, sensitivity=1.0, privacy_technique='laplace',alpha=10.0,epsilon_bar=1.0, data_kinds: List[str] = None):#Duc's code 4/10/25
        
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF], data_kinds_to_filter=data_kinds)

        self.epsilon = epsilon
        self.delta = delta
        self.gamma = gamma 
        self.sensitivity = sensitivity
        self.privacy_technique = privacy_technique
        self.alpha = alpha              #Duc's code 4/10/25
        self.epsilon_bar=epsilon_bar    #Duc's code 4/10/25

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Compute the percentile on the abs delta_W.

        Only share the params where absolute delta_W greater than
        the percentile value

        Args:
            dxo: information from client
            shareable: that the dxo belongs to
            fl_ctx: context provided by workflow

        Returns: filtered dxo
        """
       
        self.log_debug(fl_ctx, "inside filter")
        model_diff = dxo.data
        total_steps = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1)

        delta_w = np.concatenate([model_diff[name].ravel() / np.float64(total_steps) for name in sorted(model_diff)])
        self.log_info(
            fl_ctx,
            "Delta_w: Max abs: {}, Min abs: {}, Median abs: {}.".format(
                np.max(np.abs(delta_w)), np.min(np.abs(delta_w)), np.median(np.abs(delta_w))
            ),
        )

        if self.privacy_technique == 'laplace':
            # apply laplace noise
            self.log_info(fl_ctx, "Applying Laplace DP")
            delta_w = DP_Mechanisms.applyDPLaplace(delta_w, self.sensitivity, self.epsilon, self.gamma)
        elif self.privacy_technique == 'gaussian':
            self.log_info(fl_ctx, "Applying Gaussian DP")
            # apply gaussian noise
            delta_w = DP_Mechanisms.applyDPGaussian(delta_w, self.delta, self.epsilon, self.gamma)
        elif self.privacy_technique == 'RDPgaussian':   #Duc's code 4/10/25
            # apply RDPgaussian noise.                  #Duc's code 4/10/25
            self.log_info(fl_ctx, "Applying RDP Gaussian DP")
            delta_w = DP_Mechanisms.applyRDPGaussian(delta_w, self.sensitivity, self.alpha, self.epsilon_bar) #Duc's code 4/10/25
        elif self.privacy_technique == 'exponential':
            # apply exponential noise
            self.log_info(fl_ctx, "Applying Exponential DP")
            delta_w = DP_Mechanisms.applyDPExponential(delta_w, self.sensitivity, self.epsilon, self.gamma)
        elif self.privacy_technique == 'adaptive_clipping':
            # apply adaptive clipping noise
            self.log_info(fl_ctx, "Applying Adaptive Clipping DP")
            delta_w = DP_Mechanisms.applyClippingAdaptive(delta_w)
        
        # resume original format
        dp_w, _start = {}, 0
        for name in sorted(model_diff):
            if np.ndim(model_diff[name]) == 0:
                dp_w[name] = model_diff[name]
                _start += 1
                continue
            value = delta_w[_start : (_start + model_diff[name].size)]
            dp_w[name] = value.reshape(model_diff[name].shape) * np.float64(total_steps)
            _start += model_diff[name].size

        # We update the shareable weights only.  Headers are unchanged.
        dxo.data = dp_w
        return dxo

