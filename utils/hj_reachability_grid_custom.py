import functools
import itertools
import jax
from flax import struct
import jax.numpy as jnp
import numpy as np

from hj_reachability import boundary_conditions as _boundary_conditions
from hj_reachability.finite_differences import upwind_first
from hj_reachability import sets
from hj_reachability import utils

from typing import Any, Callable, Optional, Tuple, Union
from hj_reachability.boundary_conditions import BoundaryCondition

Array = Any

@struct.dataclass
class Grid:
    """Class for representing Cartesian state grids with uniform spacing in each dimension.

    Attributes:
        states: An `(N + 1)` dimensional array containing the state values at each grid location. The first `N`
            dimensions correspond to the location in the grid, while the last dimension (itself of size `N`) contains
            the state vector.
        domain: A `Box` representing the domain of grid.
        coordinate_vectors: A tuple of `N` arrays containing the discrete state values in each dimension. The `states`
            attribute is produced by `stack`ing a `meshgrid` of these coordinate vectors.
        spacings: A tuple of `N` scalars containing the grid spacing (the difference between successive elements of the
            corresponding coordinate vector) in each dimension.
        boundary_conditions: A tuple of `N` boundary conditions for each dimension. These boundary conditions are
            functions used to pad values (notably not stored in this `Grid` data structure) to implement a boundary
            condition (e.g., periodic).
    """
    states: Array
    domain: sets.Box
    coordinate_vectors: Tuple[Array, ...]
    spacings: Tuple[Array, ...]
    boundary_conditions: Tuple[BoundaryCondition, ...] = struct.field(pytree_node=False)

    @classmethod
    def from_lattice_parameters_and_boundary_conditions(
            cls,
            domain: sets.Box,
            shape: Tuple[int, ...],
            boundary_conditions: Optional[Tuple[BoundaryCondition, ...]] = None,
            periodic_dims: Optional[Union[int, Tuple[int, ...]]] = None) -> "Grid":
        """Constructs a `Grid` from a domain, shape, and boundary conditions.

        Args:
            domain: A `Box` representing the domain of grid.
            shape: A tuple of `N` integers denoting the number of discretization nodes in each dimension.
            boundary_conditions: A tuple of `N` boundary conditions for each dimension. If not provided, defaults to
                `extrapolate_away_from_zero` in each dimension, with the exception of those dimensions that appear in
                `periodic_dims` where the `periodic` boundary condition is used instead.
            periodic_dims: A single integer or tuple of integers denoting which dimensions are periodic in the case that
                the `boundary_conditions` are not explicitly provided as input to this factory method.

        Returns:
            A `Grid` constructed according to the provided specifications.
        """
        ndim = len(shape)
        if boundary_conditions is None:
            if not isinstance(periodic_dims, tuple):
                periodic_dims = (periodic_dims,)
            boundary_conditions = tuple(
                _boundary_conditions.periodic if i in periodic_dims else _boundary_conditions.extrapolate_away_from_zero
                for i in range(ndim))

        coordinate_vectors, spacings = zip(
            *(jnp.linspace(l, h, n, endpoint=bc is not _boundary_conditions.periodic, retstep=True)
              for l, h, n, bc in zip(domain.lo, domain.hi, shape, boundary_conditions)))
        states = jnp.stack(jnp.meshgrid(*coordinate_vectors, indexing="ij"), -1)

        return cls(states, domain, coordinate_vectors, spacings, boundary_conditions)

    @property
    def ndim(self) -> int:
        """Returns the dimension `N` of the grid."""
        return self.states.ndim - 1

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the grid, a tuple of `N` integers."""
        return self.states.shape[:-1]

    def upwind_grad_values(self, upwind_scheme: Callable, values: Array) -> Tuple[Array, Array]:
        """Returns `(left_grad_values, right_grad_values)`."""
        left_derivatives, right_derivatives = zip(*[
            utils.multivmap(lambda values: upwind_scheme(values, spacing, boundary_condition),
                            np.array([j
                                      for j in range(self.ndim)
                                      if j != i]))(values)
            for i, (spacing, boundary_condition) in enumerate(zip(self.spacings, self.boundary_conditions))
        ])
        return (jnp.stack(left_derivatives, -1), jnp.stack(right_derivatives, -1))

    def grad_values(self, values: Array, upwind_scheme: Optional[Callable] = None) -> Array:
        """Returns a central difference-based approximation of `grad_values`."""
        # TODO: Implement central difference schemes in `hj_reachability.finite_differences`.
        if upwind_scheme is None:
            upwind_scheme = upwind_first.first_order
        return sum(self.upwind_grad_values(upwind_scheme, values)) / 2

    def position(self, state: Array) -> Array:
        """Returns an array of `float`s corresponding to the position of `state` in the grid."""
        position = (state - self.domain.lo) / jnp.array(self.spacings)
        return jnp.where(self._is_periodic_dim, position % np.array(self.shape), position)

    def nearest_index(self, state: Array) -> Array:
        """Returns the result of rounding `self.position(state)` to the nearest grid index."""
        return jnp.round(self.position(state)).astype(jnp.int32)

    def interpolate(self, values, state):
        """Interpolates `values` (possibly multidimensional per node) defined over the grid at the given `state`."""
        position = (state - self.domain.lo) / jnp.array(self.spacings)
        index_lo = jnp.floor(position).astype(jnp.int32)
        index_hi = index_lo + 1
        weight_hi = position - index_lo
        weight_lo = 1 - weight_hi
        index_lo, index_hi = tuple(
            jnp.where(self._is_periodic_dim, index % np.array(self.shape), jnp.clip(index, 0, np.array(self.shape)))
            for index in (index_lo, index_hi))
        weight = functools.reduce(lambda x, y: x * y, jnp.ix_(*jnp.stack([weight_lo, weight_hi], -1)))
        # TODO: Double-check numerical stability here and/or switch to `tuple`s and `itertools.product` for clarity.
        return jnp.sum(
            weight[(...,) + (np.newaxis,) * (values.ndim - self.ndim)] *
            values[jnp.ix_(*jnp.stack([index_lo, index_hi], -1))], list(range(self.ndim)))

    @property
    def _is_periodic_dim(self) -> Array:
        """Returns a boolean vector indicating which dimensions (if any) are periodic."""
        return np.array([bc is _boundary_conditions.periodic for bc in self.boundary_conditions])
    
    def interpolate_timespace_batch(self, values, time_coordinate, time_states, return_grad=True, backwards_time=True):
        """
        Interpolates `values` (and optionally its gradients) at a batch of [t, x1, x2, ...] points.
        
        values: [T, *grid_shape], solved on time_coordinate (eg. output of hj_reachability.solve)
        time_coordinate: [T], evenly spaced and increasing times
        time_states: [batch, 1 + ndim] = [batch, time + space]

        Returns:
            interpolated_values: [batch]
            gradients: [batch, 1 + ndim] (using auto-diff)
        """
        
        full_ndim = self.ndim + 1  # includes time
        time_lo = time_coordinate.min()
        time_spacing = abs(time_coordinate[1] - time_coordinate[0])
        values = values[::-1] if backwards_time else values

        # Define spacing and domain bounds for time + space
        full_spacings = [time_spacing] + list(self.spacings)
        full_shape = [values.shape[0]] + list(self.shape)
        full_lo = jnp.array([time_lo] + list(self.domain.lo))

        def interpolate_timespace(state):
            """Interpolates a single state (1D array of length ndim+1)."""
            pos = (state - full_lo) / jnp.array(full_spacings)
            idx_lo = jnp.floor(pos).astype(jnp.int32)
            idx_hi = idx_lo + 1
            w_hi = pos - idx_lo
            w_lo = 1 - w_hi

            # Handle periodicity (space only)
            def clip_dim(index, dim):
                if dim == 0:  # time
                    return jnp.clip(index, 0, full_shape[0] - 1)
                elif self._is_periodic_dim[dim - 1]:
                    return index % full_shape[dim]
                else:
                    return jnp.clip(index, 0, full_shape[dim] - 1)

            idx_lo = jnp.stack([clip_dim(idx_lo[i], i) for i in range(full_ndim)])
            idx_hi = jnp.stack([clip_dim(idx_hi[i], i) for i in range(full_ndim)])

            # Build all 2^(ndim+1) corner indices
            corners = list(itertools.product([0, 1], repeat=full_ndim))
            result = 0.0

            def dynamic_index_nd(array, indices):
                """Safely index into a multi-dimensional array using a 1D vector of indices (JAX-friendly)."""
                for idx in indices:
                    array = jax.lax.dynamic_index_in_dim(array, idx, 0, keepdims=False)
                return array

            for corner in corners:
                corner_idx = jnp.stack([idx_hi[i] if c else idx_lo[i] for i, c in enumerate(corner)])
                corner_val = dynamic_index_nd(values, corner_idx)

                # Weight product
                weight = jnp.prod(jnp.array([w_hi[i] if c else w_lo[i] for i, c in enumerate(corner)]))
                result += weight * corner_val

            return result

        # Vectorize over batch
        interpolated_values = jax.vmap(interpolate_timespace)(time_states)

        if return_grad:
            # Compute gradient of interpolate_timespace using autodiff
            grad_fn = jax.jacrev(interpolate_timespace)  # gradient w.r.t. input state
            gradients = jax.vmap(grad_fn)(time_states)  # shape: (batch, ndim+1, ...)

            return interpolated_values, gradients
        else:
            return interpolated_values
        
    def grad_values_timespace(self, values: Array, time_spacing: float) -> Array:
        """
        Central difference gradient of `values` over time and space.
        values shape: [T, *grid_shape]
        returns shape: [T, *grid_shape, ndim + 1]
        """
        T = values.shape[0]

        # Time derivative
        dt_values = (
            jnp.concatenate([
                (values[1:2] - values[:1]),                     # forward diff at t=0
                (values[2:] - values[:-2]) / 2,                 # central diff
                (values[-1:] - values[-2:-1])                   # backward diff at t=T-1
            ], axis=0) / time_spacing                          # shape: [T, *grid]
        )

        # Space derivatives (vectorized over time)
        dx_values = jax.vmap(lambda v: self.grad_values(v))(values)  # shape: [T, *grid, ndim]

        # Combine along last axis
        return jnp.concatenate([dt_values[..., None], dx_values], axis=-1)  # [T, *grid, ndim+1]
    
    def interpolate_timespace_batch_opt(self, values, time_coordinate, time_states, return_grad=True, backwards_time=True):
        """
        Interpolates `values` (and optionally the gradients) at a batch of [t, x1, x2, ...] points. 
        This version is optimized for large batch sizes (and pytorch friendly).
        
        values: [T, *grid_shape]
        time_coordinate: [T], evenly spaced and increasing
        time_states: [batch, 1 + ndim] = [batch, time + space]

        Returns:
            interpolated_values: [batch]
            gradients: [batch, 1 + ndim] (using built-in central differences)
        """
        full_ndim = self.ndim + 1
        time_lo = time_coordinate.min()
        time_spacing = abs(time_coordinate[1] - time_coordinate[0])
        values = values[::-1] if backwards_time else values

        full_spacings = jnp.array([time_spacing] + list(self.spacings))
        full_lo = jnp.array([time_lo] + list(self.domain.lo))
        full_shape = [values.shape[0]] + list(self.shape)
        corner_offsets = jnp.array(list(itertools.product([0, 1], repeat=full_ndim)))  # (2**d, d)

        # Precompute gradients if requested
        if return_grad:
            grad_volume = self.grad_values_timespace(values, time_spacing)

        @jax.jit
        def interpolate_volume(volume, state):
            """Interpolate a single state from a volume (either values or grad_volume)."""
            pos = (state - full_lo) / full_spacings
            idx_lo = jnp.floor(pos).astype(jnp.int32)
            w_hi = pos - idx_lo
            w_lo = 1 - w_hi

            def get_index(i, is_hi):
                if i == 0:
                    return jnp.clip(idx_lo[0] + is_hi, 0, full_shape[0] - 1)
                else:
                    idx = idx_lo[i] + is_hi
                    if self._is_periodic_dim[i - 1]:
                        return idx % full_shape[i]
                    else:
                        return jnp.clip(idx, 0, full_shape[i] - 1)

            def get_corner(corner_bits):
                indices = jnp.array([get_index(i, corner_bits[i]) for i in range(full_ndim)])
                weight = jnp.prod(jnp.where(corner_bits, w_hi, w_lo))
                start_indices = [indices[i] for i in range(full_ndim)] + [0] * (volume.ndim - full_ndim)
                slice_sizes = [1] * full_ndim + list(volume.shape[full_ndim:])
                val = jax.lax.dynamic_slice(volume, start_indices, slice_sizes)
                return weight * val[0]

            return jnp.sum(jax.vmap(get_corner)(corner_offsets), axis=0)

        batched_interpolate_values = jax.jit(jax.vmap(lambda s: interpolate_volume(values, s)))

        if return_grad:
            batched_interpolate_grads = jax.jit(jax.vmap(lambda s: interpolate_volume(grad_volume, s)))
            vals, grads = batched_interpolate_values(time_states), batched_interpolate_grads(time_states)
            return jnp.squeeze(vals, axis=(1, 2)), jnp.squeeze(grads, axis=(1, 2))
        else:
            vals = batched_interpolate_values(time_states)
            return jnp.squeeze(vals, axis=(1, 2))
    
