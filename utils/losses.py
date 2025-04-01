import torch
from math import exp, sqrt
from scipy.special import erf
from scipy.stats import truncnorm

# uses real units
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):

    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output, diss=0.):
        
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds) - diss
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham

            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value)
                
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                # pretraining
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss

def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):

    def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask, output, diss=0.):

        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds) - diss
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham

            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
            
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
    
    return brat_hjivi_loss

def init_brt_hjivi_hopf_loss(experiment, minWith, dirichlet_loss_divisor, hopf_loss_divisor, hopf_loss_grad_divisor, hopf_loss, temporal_weighting):
    
    ## Include a Linear Differencing Term
    if hopf_loss == 'lin_val_diff':

        def brt_hjivi_loss_hopf(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output, hopf_value, learned_hopf_value, epoch, state_time):

            dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
            if experiment.dataset.dynamics.deepreach_model == 'exact':
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0

            hopf_loss = learned_hopf_value - hopf_value
            ## TODO: add thresholding for nonlin, based on conservative error?

            if torch.all(dirichlet_mask):
                # pretraining loss
                diff_constraint_hom = torch.Tensor([0])
                hopf_loss = torch.Tensor([0])

            # hopf pretraining
            elif experiment.dataset.super_pretrain: 
                diff_constraint_hom = torch.Tensor([0])

            else:
                ham = experiment.dataset.dynamics.hamiltonian(state, dvds)
                # If we are computing BRT then take min with zero
                if minWith == 'zero':
                    ham = torch.clamp(ham, max=0.0)

                diff_constraint_hom = dvdt - ham
                if minWith == 'target':
                    diff_constraint_hom = torch.max(diff_constraint_hom, value - boundary_value)

                if experiment.dataset.dynamics.deepreach_model == 'exact':
                    dirichlet = torch.Tensor([0]).cuda()

                if temporal_weighting and epoch >= experiment.total_pretrain_iters:            
                    ## flattening truncated normal params
                    a,b = 0,1
                    # mu, sigma_init, sigma_final, k = 0, 0.4, 5, 5
                    # e_perc, sigma = 0, sigma_init
                    # mu, sigma_init, sigma_final, k = 0, 0.1, 1., 2.2 # low str
                    # mu, sigma_init, sigma_final, k = 0, 0.1, 0.4, 1 # mid str
                    mu, sigma_init, sigma_final, k = 0, 0.01, 0.4, 1.8 # high str
                    e_perc = (epoch - experiment.total_pretrain_iters)/(experiment.epochs - experiment.total_pretrain_iters)
                    exp_perc = 2.718 ** (k * (e_perc - 1))
                    sigma = sigma_init * (1. - exp_perc) + sigma_final * exp_perc

                    a_p, b_p = (a - mu) / sigma, (b - mu) / sigma
                    weight = torch.tensor(truncnorm.pdf(state_time.cpu(), a_p, b_p, scale=sigma))
                    diff_constraint_hom = torch.tensor(weight).cuda() * diff_constraint_hom
                    # TODO: this still doesn't seem to do much!

            return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                    'hopf': torch.abs(hopf_loss).sum()  / hopf_loss_divisor,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
    
    elif hopf_loss == 'lin_val_grad_diff':

        def brt_hjivi_loss_hopf(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output, hopf_value, learned_hopf_value, hopf_grad, learned_hopf_grad, epoch, state_time):

            dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
            if experiment.dataset.dynamics.deepreach_model == 'exact':
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0

            hopf_loss = learned_hopf_value - hopf_value
            hopf_loss_grad = learned_hopf_grad - hopf_grad
            ## TODO: add thresholding for nonlin, based on conservative error?

            if torch.all(dirichlet_mask):
                # pretraining loss
                diff_constraint_hom = torch.Tensor([0])
                hopf_loss = torch.Tensor([0])
                hopf_loss_grad = torch.Tensor([0])

            # hopf pretraining
            elif experiment.dataset.super_pretrain: 
                diff_constraint_hom = torch.Tensor([0])

            else:
                ham = experiment.dataset.dynamics.hamiltonian(state, dvds)
                # If we are computing BRT then take min with zero
                if minWith == 'zero':
                    ham = torch.clamp(ham, max=0.0)

                diff_constraint_hom = dvdt - ham
                if minWith == 'target':
                    diff_constraint_hom = torch.max(diff_constraint_hom, value - boundary_value)

                if experiment.dataset.dynamics.deepreach_model == 'exact':
                    dirichlet = torch.Tensor([0]).cuda()

                if temporal_weighting and epoch >= experiment.total_pretrain_iters:            
                    ## flattening truncated normal params
                    a,b = 0,1
                    # mu, sigma_init, sigma_final, k = 0, 0.4, 5, 5
                    # e_perc, sigma = 0, sigma_init # fixed
                    # mu, sigma_init, sigma_final, k = 0, 0.1, 1., 2.2 # low str
                    # mu, sigma_init, sigma_final, k = 0, 0.1, 0.4, 1 # mid str
                    mu, sigma_init, sigma_final, k = 0, 0.01, 0.4, 1.8 # high str
                    e_perc = (epoch - experiment.total_pretrain_iters)/(experiment.epochs - experiment.total_pretrain_iters)
                    exp_perc = 2.718 ** (k * (e_perc - 1))
                    sigma = sigma_init * (1. - exp_perc) + sigma_final * exp_perc

                    a_p, b_p = (a - mu) / sigma, (b - mu) / sigma
                    weight = torch.tensor(truncnorm.pdf(state_time.cpu(), a_p, b_p, scale=sigma))
                    diff_constraint_hom = torch.tensor(weight).cuda() * diff_constraint_hom
                    # TODO: this still doesn't seem to do much!

            return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                    'hopf': torch.abs(hopf_loss).sum() / hopf_loss_divisor,
                    'hopf_grad': torch.abs(hopf_loss_grad).sum() / hopf_loss_grad_divisor, ## sum vs. mean vs. max ?
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    ## The following is for a self-supervised hopf grad-based loss
    # elif hopf_loss == 'lin_val_grad_diff_ssv':
        # TODO: differentiate to compute grad (J*(p) + x dot p + intg H(p, t)) (or grad^2)? (hopf grad)
        
        ## TODO: Include a Hopf Gradient term
        # p = value.backwards(state) # DxV?
        # if minWith == "zero" or minWith == "target":
        #     hgrad = grad grad (J*(p) + x dot p + intg H(p, t))
        # else:
        #     hgrad = grad (J*(p) + x dot p + intg H(p, t))

    return brt_hjivi_loss_hopf

def init_mulob_hjivi_loss(experiment, minWith, dirichlet_loss_divisor, mulob_type='BRAAT', loss_type='vanilla', 
                          dss_value_loss_1_divisor=10., dss_value_loss_2_divisor=10.,
                          dss_grad_loss_1_divisor=10., dss_grad_loss_2_divisor=10.,
                          lbss_value_loss_divisor=1., lbss_grad_loss_divisor=100.,
                          grad_super=False):
    
    grad_super = grad_super and experiment.dataset.solve_grad # must solve to use

    def mulob_hjivi_loss(state, value, grad, boundary_value, dirichlet_mask, output, bc_value_1, bc_value_2, 
                            decomposed_value_1, decomposed_value_2, decomposed_grad_1, decomposed_grad_2,
                            values_poslam, grad_poslam, values_neglam, grad_neglam):

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if experiment.dataset.dynamics.deepreach_model == 'exact' and torch.all(dirichlet_mask):
            dirichlet = output.squeeze(dim=-1)[dirichlet_mask] - 0.0

        loss_dict = {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor}

        ## Compute Supervision Losses for Decomposed Values
        if experiment.dataset.pretrained: 

            if 'augment' in loss_type:
                if not experiment.dataset.lambda_var: raise AssertionError("Augmented loss requires lambda-varying system")
                
                decomposed_value_1_loss = values_poslam - decomposed_value_1
                decomposed_value_2_loss = values_neglam - decomposed_value_2

                if grad_super:
                    decomposed_grad_1_loss = grad_poslam[...,:-1] - decomposed_grad_1
                    decomposed_grad_2_loss = grad_neglam[...,:-1] - decomposed_grad_2

                if experiment.dataset.lam_slice_super:
                    dss_value_1_weight = 1 / dss_value_loss_1_divisor
                    dss_value_2_weight = 1 / dss_value_loss_2_divisor
                    dss_grad_1_weight = 1 / dss_grad_loss_1_divisor
                    dss_grad_2_weight = 1 / dss_grad_loss_2_divisor 
                
                else: # dynamic lambda weighting
                    dss_value_1_weight = torch.nn.functional.relu(state[..., -1]) / dss_value_loss_1_divisor
                    dss_value_2_weight = torch.nn.functional.relu(-state[..., -1]) / dss_value_loss_2_divisor        
                    dss_grad_1_weight = torch.nn.functional.relu(state[..., -1]) / dss_grad_loss_1_divisor
                    dss_grad_2_weight = torch.nn.functional.relu(-state[..., -1]) / dss_grad_loss_2_divisor

                loss_dict['dss_value_1_loss'] = dss_value_1_weight * torch.abs(decomposed_value_1_loss).sum()
                loss_dict['dss_value_2_loss'] = dss_value_2_weight * torch.abs(decomposed_value_2_loss).sum()
                
                if grad_super:
                    loss_dict['dss_grad_1_loss'] = dss_grad_1_weight * torch.abs(decomposed_grad_1_loss).sum()
                    loss_dict['dss_grad_2_loss'] = dss_grad_2_weight * torch.abs(decomposed_grad_2_loss).sum()
            
            if 'deform' in loss_type:

                lbss_value_loss = value - torch.max(decomposed_value_1, decomposed_value_2)
                loss_dict['lbss_value_loss'] = lbss_value_loss / lbss_value_loss_divisor

                if grad_super:

                    if not experiment.dataset.lambda_var:
                        lbss_grad_loss = grad - torch.cat((decomposed_grad_1, decomposed_grad_2), -1)[..., torch.argmax(decomposed_value_1, decomposed_value_2)] # debug
                    else:
                        lbss_grad_loss = grad[...,:-1] - torch.cat((decomposed_grad_1, decomposed_grad_2), -1)[..., torch.argmax(decomposed_value_1, decomposed_value_2)] # debug

                    loss_dict['lbss_grad_loss'] = lbss_grad_loss / lbss_grad_loss_divisor

                # TODO?: for deformation, move dynamic weight multiplication in here for organization (scheduler can remain out)


        ## Compute PDE Loss (if no longer pretraining)
        if experiment.dataset.pretrained and (experiment.dataset.super_pretrained or not experiment.dataset.super_pretrain):
            
            dvdt, dvdx = grad[..., 0], grad[..., 1:]
            ham = experiment.dataset.dynamics.hamiltonian(state, dvdx)
            
            # Lambda shift (non-lambda) decomposed values
            if experiment.dataset.lambda_var:
                decomposed_value_1 = decomposed_value_1 - torch.nn.functional.relu(-state[..., -1])
                decomposed_value_2 = decomposed_value_2 - torch.nn.functional.relu(state[..., -1])

            ## BRT
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            ## MultiObjective Losses
            if mulob_type == 'BRAT':

                diff_constraint_hom = torch.min(torch.max(dvdt - ham, value - bc_value_1), value + bc_value_2)

            elif mulob_type == 'BRAAT':

                diff_constraint_hom = torch.min(torch.max(dvdt - ham, value - bc_value_2), 
                                                torch.max(value - bc_value_1, value - decomposed_value_2)) # not min?
                
            elif mulob_type == 'BRRT':

                diff_constraint_hom = torch.min(dvdt - ham, torch.min(
                                                torch.max(value - bc_value_2, value - decomposed_value_1), 
                                                torch.max(value - bc_value_1, value - decomposed_value_2)))
                
            loss_dict['diff_constraint_hom'] = torch.abs(diff_constraint_hom).sum()

        return loss_dict

    return mulob_hjivi_loss