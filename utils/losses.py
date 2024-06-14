import torch

# uses real units
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom, value - boundary_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss

def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.min(torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brat_hjivi_loss

def init_brt_hjivi_hopf_loss(dynamics, minWith, dirichlet_loss_divisor, hopf_loss_divisor, hopf_loss):

    ## Include a Linear Differencing Term
    if hopf_loss == 'lindiff':

        # TODO: load/solve solution? (hopf lin diff)

        # TODO: differentiate to compute grad (J*(p) + x dot p + intg H(p, t)) (or grad^2)? (hopf grad)

        def brt_hjivi_loss_hopf(state, value, dvdt, dvds, boundary_value, dirichlet_mask, hopf_value, hopf_pretrain):

            dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

            hopf_loss = value - hopf_value
            ## TODO: add thresholding for nonlin, based on conservative error?

            ## Include a Hopf Gradient term
            # p = value.backwards(state) # DxV?
            # if minWith == "zero" or minWith == "target":
            #     hgrad = grad grad (J*(p) + x dot p + intg H(p, t))
            # else:
            #     hgrad = grad (J*(p) + x dot p + intg H(p, t))

            if torch.all(dirichlet_mask):
                # pretraining loss
                diff_constraint_hom = torch.Tensor([0])
                hopf_loss = torch.Tensor([0])
            elif hopf_pretrain:
                # hopf pretraining 
                diff_constraint_hom = torch.Tensor([0])
                # hopf_loss_divisor = 1. # maybe reduced later but effective in pretraining
            else:
                ham = dynamics.hamiltonian(state, dvds)
                # If we are computing BRT then take min with zero
                if minWith == 'zero':
                    ham = torch.clamp(ham, max=0.0)

                diff_constraint_hom = dvdt - ham
                if minWith == 'target':
                    diff_constraint_hom = torch.max(diff_constraint_hom, value - boundary_value)

            return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                    'hopf': torch.abs(hopf_loss).sum()  / hopf_loss_divisor,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss_hopf