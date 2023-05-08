abstract type RBVariable{Top,Ttr} end
abstract type RBLinVariable{Top} <: RBVariable{Top,nothing} end
abstract type RBBilinVariable{Top,Ttr} <: RBVariable{Top,Ttr} end
abstract type RBLiftVariable{Top} <: RBLinVariable{Top} end

struct RBSteadyLinVariable{Top} <: RBLinVariable{Top}
  feop::ParamSteadyLinOperator{Top}
  rbspace_row::RBSpaceSteady
end

struct RBUnsteadyLinVariable{Top} <: RBLinVariable{Top}
  feop::ParamUnsteadyLinOperator{Top}
  rbspace_row::RBSpaceUnsteady
end

struct RBSteadyBilinVariable{Top,Ttr} <: RBBilinVariable{Top,Ttr}
  feop::ParamSteadyBilinOperator{Top,Ttr}
  rbspace_row::RBSpaceSteady
  rbspace_col::RBSpaceSteady
end

struct RBUnsteadyBilinVariable{Top,Ttr} <: RBBilinVariable{Top,Ttr}
  feop::ParamUnsteadyBilinOperator{Top,Ttr}
  rbspace_row::RBSpaceUnsteady
  rbspace_col::RBSpaceUnsteady
end

struct RBSteadyLiftVariable{Top} <: RBLiftVariable{Top}
  feop::ParamSteadyLiftOperator{Top}
  rbspace_row::RBSpaceSteady

  function RBSteadyLiftVariable(
    feop::ParamSteadyLiftOperator{Top},
    rbspace_row::RBSpaceSteady) where {Top}
    new{Top}(feop,rbspace_row)
  end
end

struct RBUnsteadyLiftVariable{Top} <: RBLiftVariable{Top}
  feop::ParamUnsteadyLiftOperator{Top}
  rbspace_row::RBSpaceUnsteady

  function RBUnsteadyLiftVariable(
    feop::ParamUnsteadyLiftOperator{Top},
    rbspace_row::RBSpaceUnsteady) where {Top}
    new{Top}(feop,rbspace_row)
  end
end

function RBVariable(
  feop::ParamSteadyLinOperator{Top},
  rbspace_row::RBSpaceSteady) where Top

  RBSteadyLinVariable{Top}(feop,rbspace_row)
end

function RBVariable(
  feop::ParamUnsteadyLinOperator{Top},
  rbspace_row::RBSpaceUnsteady) where Top

  RBUnsteadyLinVariable{Top}(feop,rbspace_row)
end

function RBVariable(
  feop::ParamSteadyBilinOperator{Top,Ttr},
  rbspace_row::RBSpaceSteady,
  rbspace_col::RBSpaceSteady) where {Top,Ttr}

  RBSteadyBilinVariable{Top,Ttr}(feop,rbspace_row,rbspace_col)
end

function RBVariable(
  feop::ParamUnsteadyBilinOperator{Top,Ttr},
  rbspace_row::RBSpaceUnsteady,
  rbspace_col::RBSpaceUnsteady) where {Top,Ttr}

  RBUnsteadyBilinVariable{Top,Ttr}(feop,rbspace_row,rbspace_col)
end

function RBLiftVariable(op::RBSteadyBilinVariable)
  feop = ParamLiftOperator(op.feop)
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpaceSteady(get_id(rbs)*:_lift,get_basis_space(rbs))
  RBSteadyLiftVariable(feop,rbs_lift)
end

function RBLiftVariable(op::RBUnsteadyBilinVariable)
  feop = ParamLiftOperator(op.feop)
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpaceUnsteady(get_id(rbs)*:_lift,get_basis_space(rbs),get_basis_time(rbs))
  RBUnsteadyLiftVariable(feop,rbs_lift)
end

const RBSteadyVariable{Top,Ttr} =
  Union{RBSteadyLinVariable{Top},RBSteadyBilinVariable{Top,Ttr},RBSteadyLiftVariable{Top}}

const RBUnsteadyVariable{Top,Ttr} =
  Union{RBUnsteadyLinVariable{Top},RBUnsteadyBilinVariable{Top,Ttr},RBUnsteadyLiftVariable{Top}}

get_param_function(rbop::RBVariable) = get_param_function(rbop.feop)

get_param_fefunction(rbop::RBVariable) = get_param_fefunction(rbop.feop)

get_rbspace_row(rbop::RBVariable) = rbop.rbspace_row

get_rbspace_col(rbop::RBBilinVariable) = rbop.rbspace_col

get_id(rbop::RBVariable) = get_id(rbop.feop)

get_basis_space_row(rbop::RBVariable) = get_basis_space(get_rbspace_row(rbop))

get_basis_space_col(rbop::RBVariable) = get_basis_space(get_rbspace_col(rbop))

Gridap.FESpaces.get_test(rbop::RBVariable) = get_test(rbop.feop)

Gridap.FESpaces.get_trial(rbop::RBVariable) = get_trial(rbop.feop)

get_dimension(rbop::RBVariable) = get_dimension(rbop.feop)

get_basis_time_row(rbop::RBVariable) = get_basis_time(get_rbspace_row(rbop))

get_basis_time_col(rbop::RBVariable) = get_basis_time(get_rbspace_col(rbop))

get_Nt(rbop::RBVariable) = get_Nt(rbop.rbspace_row)

get_ns_row(rbop::RBVariable) = get_ns(get_rbspace_row(rbop))

get_nt_row(rbop::RBUnsteadyVariable) = get_nt(get_rbspace_row(rbop))

get_ns_col(rbop::RBVariable) = 1

get_nt_col(rbop::RBUnsteadyVariable) = 1

get_ns_col(rbop::RBBilinVariable) = get_ns(get_rbspace_col(rbop))

get_nt_col(rbop::RBUnsteadyBilinVariable) = get_nt(get_rbspace_row(rbop))

get_nrows(rbop::RBSteadyVariable) = get_ns_row(rbop)

get_nrows(rbop::RBUnsteadyVariable) = get_ns_row(rbop)*get_nt_row(rbop)

get_ncols(rbop::RBSteadyVariable) = get_ns_col(rbop)

get_ncols(rbop::RBUnsteadyVariable) = get_ns_col(rbop)*get_nt_col(rbop)

function Gridap.FESpaces.get_cell_dof_ids(
  rbop::RBVariable,
  trian::Triangulation)
  get_cell_dof_ids(rbop.feop,trian)
end

get_assembler(op::RBVariable) = get_assembler(op.feop)

assemble_affine_quantity(op::RBVariable,args...) = assemble_affine_quantity(op.feop)

get_dirichlet_function(op::RBVariable) = get_dirichlet_function(op.feop)

get_pspace(op::RBVariable) = get_pspace(op.feop)

realization(op::RBVariable) = realization(get_pspace(op))

get_time_info(op::RBUnsteadyLinVariable) = get_time_info(op.feop)

get_time_info(op::RBUnsteadyBilinVariable) = get_time_info(op.feop)

get_dt(op::RBVariable) = get_dt(op.feop)

get_Nt(op::RBVariable) = get_Nt(op.feop)

get_θ(op::RBVariable) = get_θ(op.feop)

get_times(op::RBVariable) = get_times(op.feop)

assemble_fe_snaps(op::RBVariable,args...;kwargs...) = assemble_fe_snaps(op.feop,args...;kwargs...)

assemble_functional_snaps(op::RBVariable,args...) = assemble_functional_snaps(op.feop,args...)

function rb_space_projection(
  op::RBLinVariable,
  vec::AbstractArray)

  rbrow = get_rbspace_row(op)
  reshape(rb_space_projection(rbrow,vec),:,1)
end

function rb_space_projection(
  op::RBBilinVariable,
  mat::AbstractArray)

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  reshape(rb_space_projection(rbrow,rbcol,mat),:,1)
end

function rb_space_projection(op::RBVariable)
  mv = assemble_affine_quantity(op)
  rb_space_projection(op,mv)
end

function rb_time_projection(
  op::RBLinVariable,
  vec::AbstractArray)

  rbrow = get_rbspace_row(op)
  rb_time_projection(rbrow,vec)
end

function rb_time_projection(
  op::RBBilinVariable,
  mat::AbstractArray;
  idx_forwards=1:size(mv,1),
  idx_backwards=1:size(mv,1))

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  rb_time_projection(rbrow,rbcol,mat;
    idx_forwards=idx_forwards,idx_backwards=idx_backwards)
end

function rb_spacetime_projection(
  op::RBLinVariable,
  vec::AbstractArray)

  rbrow = get_rbspace_row(op)
  rb_spacetime_projection(rbrow,vec)
end

function rb_spacetime_projection(
  op::RBBilinVariable,
  mat::AbstractArray;
  idx_forwards=1:size(mv,1),
  idx_backwards=1:size(mv,1))

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  rb_spacetime_projection(rbrow,rbcol,mat;
    idx_forwards=idx_forwards,idx_backwards=idx_backwards)
end

function Gridap.FESpaces.FEFunction(
  op::RBSteadyLinVariable,
  u::AbstractVector,
  ::Param)

  FEFunction(get_test(op),u)
end

function Gridap.FESpaces.FEFunction(
  op::RBUnsteadyLinVariable,
  u::AbstractMatrix,
  ::Param)

  times = get_times(op)
  n(tn) = findall(x->x == tn,times)[1]
  tn -> FEFunction(get_test(op),u[:,n(tn)])
end

function Gridap.FESpaces.FEFunction(
  op::Union{RBSteadyBilinVariable,RBSteadyLiftVariable},
  u::AbstractVector,
  μ::Param)

  FEFunction(get_trial(op)(μ),u)
end

function Gridap.FESpaces.FEFunction(
  op::Union{RBUnsteadyBilinVariable,RBUnsteadyLiftVariable},
  u::AbstractMatrix,
  μ::Param)

  times = get_times(op)
  n(tn) = findall(x->x == tn,times)[1]
  tn -> FEFunction(get_trial(op)(μ,tn),u[:,n(tn)])
end
