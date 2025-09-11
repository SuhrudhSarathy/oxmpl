// Copyright (c) 2025 Junior Sundar
//
// SPDX-License-Identifier: BSD-3-Clause

use std::any::Any;

pub use crate::base::states::{
    compound_state::CompoundState, real_vector_state::RealVectorState, so2_state::SO2State,
    so3_state::SO3State,
};

pub trait DynClone {
    fn clone_box(&self) -> Box<dyn State>;
}

impl<T> DynClone for T
where
    T: State + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }
}

/// A marker trait for all state types in the planning library.
///
/// A `State` represents a single point, configuration, or snapshot of the system
/// being planned for.
///
/// Supertrait bounds:
/// - `DynClone`: States must be copyable as Dyn for runtime polymorphism.
///
/// > [!NOTE] (for self)
/// > A trait is not dyn-compatible if any of its methods return Self — unless it has a `where Self: Sized` bound.
pub trait State: DynClone + Any + 'static {}

impl Clone for Box<dyn State> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// An enum that holds any of the library's built-in `State` types.
///
/// This enum is the counterpart to `StateSpaceVariant` and is the core of the "enum dispatch"
/// pattern. It allows a `CompoundState` to hold a heterogeneous collection of different concrete
/// state types in a type-safe way.
///
/// The `Custom` variant provides an "escape hatch" for users to include their own `State`
/// implementations.
///
/// # Examples
///
/// ```
/// use oxmpl::base::state::{CompoundState, RealVectorState, SO2State, StateVariant};
///
/// let pos_variant = StateVariant::RealVector(RealVectorState::new(vec![1.0, 2.0]));
/// let rot_variant = StateVariant::SO2(SO2State::new(std::f64::consts::PI));
///
/// let combined_state = CompoundState {
///     components: vec![pos_variant, rot_variant],
/// };
/// ```
#[derive(Clone)]
pub enum StateVariant {
    RealVector(RealVectorState),
    SO2(SO2State),
    SO3(SO3State),
    // -- Add more as they come
    Custom(Box<dyn State>),
}
