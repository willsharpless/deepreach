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

        hjb_residual = dvdt - (dynamics.hamiltonian(state, dvds) - diss)
        loss_priority_residual = hjb_residual + value
        hjb_loss_perc = ((loss_priority_residual >= -avoid_value) & (loss_priority_residual <= reach_value)).sum()/value.shape[-1]

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
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                        'hjb_loss_perc':hjb_loss_perc, 
                        'hjb_residual_abs_avg':hjb_residual.abs().mean(), 'hjb_residual_std':hjb_residual.abs().std(),
                        'BRAT_priority_residual_avg':loss_priority_residual.mean(), 'BRAT_priority_residual_std':loss_priority_residual.std(), 
                        'value_avg':value.mean(), 'value_std':value.std(),
                        'value_1_avg':reach_value.mean(), 'value_1_std':reach_value.std(),
                        'value_2_avg':(-avoid_value).mean(), 'value_2_std':(-avoid_value).std(),
                        }
        
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                'hjb_loss_perc':hjb_loss_perc, 
                'hjb_residual_abs_avg':hjb_residual.abs().mean(), 'hjb_residual_std':hjb_residual.abs().std(),
                'BRAT_priority_residual_avg':loss_priority_residual.mean(), 'BRAT_priority_residual_std':loss_priority_residual.std(), 
                'value_avg':value.mean(), 'value_std':value.std(),
                'value_1_avg':reach_value.mean(), 'value_1_std':reach_value.std(),
                'value_2_avg':(-avoid_value).mean(), 'value_2_std':(-avoid_value).std(),
                }
    
    return brat_hjivi_loss

def init_mulob_hjivi_loss(experiment, minWith, dirichlet_loss_divisor, mulob_type='BRAAT', loss_type='vanilla', 
                          dss_value_loss_1_divisor=10., dss_value_loss_2_divisor=10.,
                          dss_grad_loss_1_divisor=10., dss_grad_loss_2_divisor=10.,
                          lbss_value_loss_divisor=1., lbss_grad_loss_divisor=100.,
                          ):
    
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
                
                ## Value Supervision (batch subset if lambda slice supervision)
                decomposed_value_1_subset = decomposed_value_1[:, :experiment.dataset.numpoints_super] #if experiment.dataset.lam_slice_super else decomposed_value_1
                decomposed_value_2_subset = decomposed_value_2[:, :experiment.dataset.numpoints_super] #if experiment.dataset.lam_slice_super else decomposed_value_2

                decomposed_value_1_loss = values_poslam - decomposed_value_1_subset
                decomposed_value_2_loss = values_neglam + decomposed_value_2_subset if hasattr(experiment.dataset.dynamics, "avoid_fn") else values_neglam - decomposed_value_2_subset

                ## Gradient Supervision
                if experiment.dataset.grad_super:

                    decomposed_grad_1_subset = decomposed_grad_1[:, :experiment.dataset.numpoints_super, :] #if experiment.dataset.lam_slice_super else decomposed_grad_1
                    decomposed_grad_2_subset = decomposed_grad_2[:, :experiment.dataset.numpoints_super, :] #if experiment.dataset.lam_slice_super else decomposed_grad_2

                    if experiment.dataset.grad_super_time:
                        decomposed_grad_1_loss = grad_poslam[...,:-1] - decomposed_grad_1_subset
                        decomposed_grad_2_loss = grad_neglam[...,:-1] + decomposed_grad_2_subset if hasattr(experiment.dataset.dynamics, "avoid_fn") else grad_neglam[...,:-1] - decomposed_grad_2_subset
                    else:
                        decomposed_grad_1_loss = grad_poslam[...,1:-1] - decomposed_grad_1_subset[..., 1:]
                        decomposed_grad_2_loss = grad_neglam[...,1:-1] + decomposed_grad_2_subset[..., 1:] if hasattr(experiment.dataset.dynamics, "avoid_fn") else grad_neglam[...,1:-1] - decomposed_grad_2_subset[..., 1:]

                if experiment.dataset.lam_slice_super:
                    dss_value_1_weight = 1 / dss_value_loss_1_divisor
                    dss_value_2_weight = 1 / dss_value_loss_2_divisor
                    dss_grad_1_weight = 1 / dss_grad_loss_1_divisor
                    dss_grad_2_weight = 1 / dss_grad_loss_2_divisor 
                
                else: ## Free lambda supervision weighting (non-slice supervision)
                    dss_value_1_weight = torch.nn.functional.relu(state[:, :experiment.dataset.numpoints_super, -1]) / dss_value_loss_1_divisor
                    dss_value_2_weight = torch.nn.functional.relu(-state[:, :experiment.dataset.numpoints_super, -1]) / dss_value_loss_2_divisor        
                    dss_grad_1_weight = torch.nn.functional.relu(state[:, :experiment.dataset.numpoints_super, -1]) / dss_grad_loss_1_divisor
                    dss_grad_2_weight = torch.nn.functional.relu(-state[:, :experiment.dataset.numpoints_super, -1]) / dss_grad_loss_2_divisor

                loss_dict['dss_value_1_loss'] = (dss_value_1_weight * torch.abs(decomposed_value_1_loss)).sum()
                loss_dict['dss_value_2_loss'] = (dss_value_2_weight * torch.abs(decomposed_value_2_loss)).sum()
                
                if experiment.dataset.grad_super:
                    loss_dict['dss_grad_1_loss'] = (dss_grad_1_weight * torch.abs(decomposed_grad_1_loss).sum(-1)).sum()
                    loss_dict['dss_grad_2_loss'] = (dss_grad_2_weight * torch.abs(decomposed_grad_2_loss).sum(-1)).sum()
            
            if 'deform' in loss_type:

                lbss_value_loss = value - torch.max(decomposed_value_1, decomposed_value_2)
                loss_dict['lbss_value_loss'] = lbss_value_loss / lbss_value_loss_divisor

                if experiment.dataset.grad_super:
                    if not experiment.dataset.lambda_var: #FIXME need to debug surely, also pre-batch subset
                        if experiment.dataset.grad_super_time:
                            lbss_grad_loss = grad - torch.cat((decomposed_grad_1, decomposed_grad_2), -1)[..., torch.argmax(decomposed_value_1, decomposed_value_2)] # debug
                        else:
                            lbss_grad_loss = grad[..., 1:] - torch.cat((decomposed_grad_1, decomposed_grad_2), -1)[..., torch.argmax(decomposed_value_1, decomposed_value_2)][..., 1:] # debug
                    else:
                        if experiment.dataset.grad_super_time:
                            lbss_grad_loss = grad[...,:-1] - torch.cat((decomposed_grad_1, decomposed_grad_2), -1)[..., torch.argmax(decomposed_value_1, decomposed_value_2)] # debug
                        else:
                            lbss_grad_loss = grad[...,1:-1] - torch.cat((decomposed_grad_1, decomposed_grad_2), -1)[..., torch.argmax(decomposed_value_1, decomposed_value_2)][..., 1:] # debug
                            
                    loss_dict['lbss_grad_loss'] = lbss_grad_loss / lbss_grad_loss_divisor

                # TODO?: for deformation, move dynamic weight multiplication in here for organization (scheduler can remain out)

        ## Compute PDE Loss (if no longer pretraining)
        if experiment.dataset.pretrained and (experiment.dataset.super_pretrained or not experiment.dataset.super_pretrain):
            
            dvdt, dvdx = grad[..., 0], grad[..., 1:]
            ham = experiment.dataset.dynamics.hamiltonian(state, dvdx)
            
            hjb_residual = dvdt - ham
            loss_priority_residual = hjb_residual + value
            
            # Lambda shift (non-lambda) decomposed values
            if experiment.dataset.lambda_var:
                decomposed_value_1 = decomposed_value_1 - torch.nn.functional.relu(-state[..., -1])
                decomposed_value_2 = decomposed_value_2 + torch.nn.functional.relu(state[..., -1])

            ## MultiObjective Losses
            if mulob_type == 'BRAT':

                diff_constraint_hom = torch.min(torch.max(hjb_residual, value - bc_value_1), value + bc_value_2)

                loss_dict['hjb_loss_perc'] = ((loss_priority_residual >= decomposed_value_2) & (loss_priority_residual <= decomposed_value_1)).sum()/value.shape[-1]

            elif mulob_type == 'BRAAT':

                diff_constraint_hom = torch.max(torch.min(hjb_residual, value + bc_value_2), 
                                                torch.min(value - bc_value_1, value + decomposed_value_2))
                
                # TODO: loss_dict['hjb_loss_perc'] = ((loss_priority_residual >= decomposed_value_2) & (loss_priority_residual <= decomposed_value_1)).sum()/value.shape[-1]
                
            elif mulob_type == 'BRRT':

                # diff_constraint_hom = torch.min(dvdt - ham, torch.min(
                #                                 torch.max(value - bc_value_2, value - decomposed_value_1), 
                #                                 torch.max(value - bc_value_1, value - decomposed_value_2)))
                
                diff_constraint_hom = torch.max(hjb_residual, torch.max(
                                                torch.min(value - bc_value_2, value - decomposed_value_1), 
                                                torch.min(value - bc_value_1, value - decomposed_value_2)))
                
                # TODO: loss_dict['hjb_loss_perc'] = ((loss_priority_residual >= decomposed_value_2) & (loss_priority_residual <= decomposed_value_1)).sum()/value.shape[-1]
                
            loss_dict['diff_constraint_hom'] = torch.abs(diff_constraint_hom).sum()

            loss_dict['hjb_residual_avg'] = hjb_residual.mean()
            loss_dict['hjb_residual_std'] = hjb_residual.std()
            loss_dict['value_avg'] = value.mean()
            loss_dict['value_std'] = value.std()
            loss_dict['value_1_avg'] = decomposed_value_1.mean()
            loss_dict['value_1_std'] = decomposed_value_1.std()
            loss_dict['value_2_avg'] = decomposed_value_2.mean()
            loss_dict['value_2_std'] = decomposed_value_2.std()

        return loss_dict

    return mulob_hjivi_loss