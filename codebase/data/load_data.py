 

#!/usr/bin/env python3
 
from ..models.simulate import simulate_data
 

def load_simulate_data(model, hparams, batch_size, n_batch, beta):
    """ 
    To speed up computation, only supports 1-batch. 
     """

    simulate_loader = simulate_data(model,
                                    batch_size,
                                    n_batch,
                                    hparams,
                                    beta=beta)

    return simulate_loader
