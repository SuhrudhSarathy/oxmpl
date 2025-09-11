// Copyright (c) 2025 Junior Sundar
//
// SPDX-License-Identifier: BSD-3-Clause

use crate::base::state::{State, StateVariant};

#[derive(Clone)]
pub struct CompoundState {
    pub components: Vec<StateVariant>,
}

impl State for CompoundState {}
