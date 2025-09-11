// Copyright (c) 2025 Junior Sundar
//
// SPDX-License-Identifier: BSD-3-Clause

use rand::Rng;

pub use crate::base::spaces::{
    any_state_space::AnyStateSpace, compound_state_space::CompoundStateSpace,
    real_vector_state_space::RealVectorStateSpace, so2_state_space::SO2StateSpace,
    so3_state_space::SO3StateSpace,
};

use crate::base::{
    error::StateSamplingError,
    state::{State, StateVariant},
};

/// Defines a space in which planning can be performed.
///
/// A `StateSpace` represents the manifold where states exist. It defines the properties and
/// operations applicable to that space as a whole, such as how to measure distance, how to
/// interpolate between states, and how to generate new states.
///
/// This trait is generic and can be implemented for various types of spaces, like N-dimensional
/// Euclidean vectors (`RealVectorStateSpace`) or 2D rotations (`SO2StateSpace`). Planners are
/// written to be generic over this trait, allowing them to solve problems in any space that
/// implements these fundamental operations.
///
/// # Examples
///
/// ```
/// use std::f64;
/// use oxmpl::base::state::State;
/// use oxmpl::base::space::StateSpace;
/// use oxmpl::base::error::StateSamplingError;
/// use rand::Rng;
///
/// #[derive(Debug, Clone, PartialEq)]
/// struct Point1D {
///     x: f64,
/// }
/// impl State for Point1D {}
///
/// struct LineSegmentSpace {
///     bounds: (f64, f64),
/// }
///
/// impl StateSpace for LineSegmentSpace {
///     type StateType = Point1D;
///
///     fn distance(&self, state1: &Self::StateType, state2: &Self::StateType) -> f64 {
///         (state1.x - state2.x).abs()
///     }
///
///     fn interpolate(&self, from: &Self::StateType, to: &Self::StateType, t: f64, state: &mut Self::StateType) {
///         state.x = from.x + (to.x - from.x) * t;
///     }
///
///     fn enforce_bounds(&self, state: &mut Self::StateType) {
///         state.x = state.x.clamp(self.bounds.0, self.bounds.1);
///     }
///
///     fn satisfies_bounds(&self, state: &Self::StateType) -> bool {
///         state.x >= self.bounds.0 && state.x <= self.bounds.1
///     }
///
///     fn sample_uniform(&self, rng: &mut impl Rng) -> Result<Self::StateType, StateSamplingError> {
///         Ok(Point1D { x: rng.gen_range(self.bounds.0..self.bounds.1) })
///     }
///
///     fn get_longest_valid_segment_length(&self) -> f64 {
///         (self.bounds.1 - self.bounds.0) * 0.05
///     }
/// }
///
/// let space = LineSegmentSpace { bounds: (0.0, 10.0) };
/// let mut rng = rand::rng();
/// let random_state = space.sample_uniform(&mut rng).unwrap();
///
/// assert!(space.satisfies_bounds(&random_state));
/// assert_eq!(space.get_longest_valid_segment_length(), 0.5);
/// ```
pub trait StateSpace {
    /// StateType defines what is acceptable in current StateSpace
    type StateType: State;

    /// Find distance between current state1 and target state2.
    ///
    /// The distance metric is specific to the topology of the space. For example, a
    /// `RealVectorStateSpace` would use Euclidean distance, while an `SO2StateSpace` would compute
    /// the shortest angle on a circle.
    ///
    /// # Parameters
    /// * `state1` - The first state.
    /// * `state2` - The second state.
    fn distance(&self, state1: &Self::StateType, state2: &Self::StateType) -> f64;

    /// Find state interpolated between `from` and `to` states given 0<=`t`<=1.
    ///
    /// The resulting state is a point on the path between `from` and `to`, determined by the
    /// interpolation parameter `t`. The path is assumed to be a straight line.
    ///
    /// # Parameters
    /// * `from` - The starting state for interpolation.
    /// * `to` - The ending state for interpolation.
    /// * `t` - The interpolation factor.
    /// * `state` - A mutable reference to a state that will be updated with the result.
    fn interpolate(
        &self,
        from: &Self::StateType,
        to: &Self::StateType,
        t: f64,
        state: &mut Self::StateType,
    );

    /// Modifies the given state to ensure it conforms to the space's defined bounds.
    ///
    /// A `RealVectorStateSpace` might clamp the values of the state to its min/max bounds.
    ///
    /// While an `SO2StateSpace` would normalise an angle to range (e.g., `[-PI, PI)`). This method
    /// modifies the state in-place.
    fn enforce_bounds(&self, state: &mut Self::StateType);

    /// Checks if a state is within the valid bounds of this space.
    ///
    /// This method only checks against the fundamental boundaries of the space definition. It does
    /// *not* check for things like collisions, which is the job of a `StateValidityChecker`.
    ///
    /// # Returns
    /// Returns `true` if the state is within bounds, `false` otherwise.
    fn satisfies_bounds(&self, state: &Self::StateType) -> bool;

    /// Generates a state uniformly at random from the entire state space.
    ///
    /// This method relies on the space having well-defined, finite bounds.
    ///
    /// # Parameters
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Errors
    /// Returns a `StateSamplingError::UnboundedDimension` if the space is unbounded
    /// in any dimension, as uniform sampling from an infinite domain is not possible.
    fn sample_uniform(&self, rng: &mut impl Rng) -> Result<Self::StateType, StateSamplingError>;

    /// Gets the length of the longest segment that can be assumed valid.
    ///
    /// This is a heuristic used to determine the resolution for motion validation. A smaller value
    /// means motions are checked more frequently.
    fn get_longest_valid_segment_length(&self) -> f64;
}

/// An enum that holds any of the library's built-in `StateSpace` types.
///
/// Using "enum dispatch" to provide a form of compile-time polymorphism. It allows a
/// `CompoundStateSpace` to hold a collection of different state spaces without relying entirely on
/// `dyn Trait` objects.
///
/// The `Custom` variant provides an "escape hatch" for users to provide their own `StateSpace`
/// implementations.
pub enum StateSpaceVariant {
    RealVector(RealVectorStateSpace),
    SO2(SO2StateSpace),
    SO3(SO3StateSpace),
    // -- Add more as they come
    /// A fallback for user-defined, dynamically dispatched state spaces.
    Custom(Box<dyn AnyStateSpace>),
}

/// Implements the dispatch logic for all `StateSpace` methods.
///
/// Each method in this block uses a `match` statement to delegate the call to the appropriate
/// method on the concrete `StateSpace` type held by the enum variant.
impl StateSpaceVariant {
    /// Dispatches the `distance` call to the appropriate concrete implementation.
    ///
    /// # Panics
    /// Panics if the `StateVariant` types do not match the `StateSpaceVariant` type.
    pub fn distance(&self, s1: &StateVariant, s2: &StateVariant) -> f64 {
        match (self, s1, s2) {
            (
                Self::RealVector(space),
                StateVariant::RealVector(state1),
                StateVariant::RealVector(state2),
            ) => space.distance(state1, state2),
            (Self::SO2(space), StateVariant::SO2(state1), StateVariant::SO2(state2)) => {
                space.distance(state1, state2)
            }
            (Self::SO3(space), StateVariant::SO3(state1), StateVariant::SO3(state2)) => {
                space.distance(state1, state2)
            }
            // -- Add more as they come
            (Self::Custom(space), StateVariant::Custom(state1), StateVariant::Custom(state2)) => {
                space.distance_dyn(&**state1, &**state2)
            }
            _ => panic!("Mismatched StateSpace and State variants in distance call."),
        }
    }

    /// Dispatches the `interpolate` call to the appropriate concrete implementation.
    ///
    /// # Panics
    /// Panics if the `StateVariant` types do not match the `StateSpaceVariant` type.
    pub fn interpolate(
        &self,
        from: &StateVariant,
        to: &StateVariant,
        t: f64,
        out: &mut StateVariant,
    ) {
        match (self, from, to, out) {
            (
                Self::RealVector(space),
                StateVariant::RealVector(f),
                StateVariant::RealVector(t_s),
                StateVariant::RealVector(o),
            ) => space.interpolate(f, t_s, t, o),
            (
                Self::SO2(space),
                StateVariant::SO2(f),
                StateVariant::SO2(t_s),
                StateVariant::SO2(o),
            ) => space.interpolate(f, t_s, t, o),
            (
                Self::SO3(space),
                StateVariant::SO3(f),
                StateVariant::SO3(t_s),
                StateVariant::SO3(o),
            ) => space.interpolate(f, t_s, t, o),
            // -- Add more as they come
            (
                Self::Custom(space),
                StateVariant::Custom(f),
                StateVariant::Custom(t_s),
                StateVariant::Custom(o),
            ) => space.interpolate_dyn(&**f, &**t_s, t, &mut **o),
            _ => panic!("Mismatched StateSpace and State variants in interpolate call."),
        }
    }

    /// Dispatches the `enforce_bounds` call to the appropriate concrete implementation.
    pub fn enforce_bounds(&self, state: &mut StateVariant) {
        match (self, state) {
            (Self::RealVector(space), StateVariant::RealVector(st)) => space.enforce_bounds(st),
            (Self::SO2(space), StateVariant::SO2(st)) => space.enforce_bounds(st),
            (Self::SO3(space), StateVariant::SO3(st)) => space.enforce_bounds(st),
            // -- Add more as they come
            (Self::Custom(space), StateVariant::Custom(st)) => space.enforce_bounds_dyn(&mut **st),
            _ => panic!("Mismatched StateSpace and State variants in enforce_bounds call."),
        }
    }

    /// Dispatches the `satisfies_bounds` call to the appropriate concrete implementation.
    pub fn satisfies_bounds(&self, state: &StateVariant) -> bool {
        match (self, state) {
            (Self::RealVector(space), StateVariant::RealVector(st)) => space.satisfies_bounds(st),
            (Self::SO2(space), StateVariant::SO2(st)) => space.satisfies_bounds(st),
            (Self::SO3(space), StateVariant::SO3(st)) => space.satisfies_bounds(st),
            // -- Add more as they come
            (Self::Custom(space), StateVariant::Custom(st)) => space.satisfies_bounds_dyn(&**st),
            _ => panic!("Mismatched StateSpace and State variants in satisfies_bounds call."),
        }
    }

    /// Dispatches the `sample_uniform` call, handling both generic and `dyn` cases.
    pub fn sample_uniform(&self, rng: &mut impl Rng) -> Result<StateVariant, StateSamplingError> {
        match self {
            Self::RealVector(space) => Ok(StateVariant::RealVector(space.sample_uniform(rng)?)),
            Self::SO2(space) => Ok(StateVariant::SO2(space.sample_uniform(rng)?)),
            Self::SO3(space) => Ok(StateVariant::SO3(space.sample_uniform(rng)?)),
            // -- Add more as they come
            Self::Custom(space) => Ok(StateVariant::Custom(space.sample_uniform_dyn(rng)?)),
        }
    }

    /// Dispatches the `get_longest_valid_segment_length` call.
    pub fn get_longest_valid_segment_length(&self) -> f64 {
        match self {
            Self::RealVector(space) => space.get_longest_valid_segment_length(),
            Self::SO2(space) => space.get_longest_valid_segment_length(),
            Self::SO3(space) => space.get_longest_valid_segment_length(),
            // -- Add more as they come
            Self::Custom(space) => space.get_longest_valid_segment_length_dyn(),
        }
    }
}
