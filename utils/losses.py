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

## Include a Linear Differencing Term
def init_brt_hjivi_loss_lindif(dynamics, minWith, dirichlet_loss_divisor):

    # load solution? solve solution?

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

        # lin_diff = value - linear_value(state)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss

## Include a Hopf Gradient term
def init_brt_hjivi_loss_hopfgrad(dynamics, minWith, dirichlet_loss_divisor):

    # differentiate to compute grad (J*(p) + x dot p + intg H(p, t)) (or grad^2)

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

        # p = value.backwards(state) # DxV?
        # if minWith == "zero" or minWith == "target":
        #     hgrad = grad grad (J*(p) + x dot p + intg H(p, t))
        # else:
        #     hgrad = grad (J*(p) + x dot p + intg H(p, t))

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss