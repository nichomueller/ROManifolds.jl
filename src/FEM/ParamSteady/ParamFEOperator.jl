"""
    abstract type ParamFEOperator{T<:ParamOperatorType} <: FEOperator end

Parametric extension of a [`FEOperator`](@ref) in [`Gridap`](@ref). Compared to
a standard FEOperator, there are the following novelties:

- a ParamSpace is provided, so that parametric realizations can be extracted
  directly from the ParamFEOperator
- an AbstractIndexMap is provided, so that a nonstandard indexing strategy can
  take place when dealing with FEFunctions
- a function representing a norm matrix is provided, so that errors in the
  desired norm can be automatically computed

Subtypes:

- [`ParamFEOpFromWeakForm`](@ref)
- [`ParamSaddlePointFEOp`](@ref)
- [`ParamFEOperatorWithTrian`](@ref)
- [`GenericLinearNonlinearParamFEOperator`](@ref)

"""
abstract type ParamFEOperator{T<:ParamOperatorType} <: FEOperator end

function FESpaces.get_algebraic_operator(op::ParamFEOperator)
  ParamOpFromFEOp(op)
end

function allocate_pfeopcache(op::ParamFEOperator,r::ParamRealization,u::AbstractVector)
  nothing
end

function update_pfeopcache!(tfeopcache,op::ParamFEOperator,r::ParamRealization)
  tfeopcache
end

ParamDataStructures.realization(op::ParamFEOperator;kwargs...) = @abstractmethod

function ParamFESpaces.get_param_assembler(op::ParamFEOperator,r::ParamRealization)
  get_param_assembler(get_assembler(op),r)
end

IndexMaps.get_index_map(op::ParamFEOperator) = @abstractmethod
get_vector_index_map(op::ParamFEOperator) = get_vector_index_map(get_index_map(op))
get_matrix_index_map(op::ParamFEOperator) = get_matrix_index_map(get_index_map(op))

get_induced_norm(op::ParamFEOperator) = @abstractmethod

function assemble_norm_matrix(op::ParamFEOperator)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  inorm = get_induced_norm(op)
  assemble_norm_matrix(inorm,trial,test)
end

get_coupling(op::ParamFEOperator) = @abstractmethod

function assemble_coupling_matrix(op::ParamFEOperator)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  c = get_coupling(op)
  assemble_matrix(c,trial,test)
end

get_linear_operator(op::ParamFEOperator) = @abstractmethod
get_nonlinear_operator(op::ParamFEOperator) = @abstractmethod

"""
    struct ParamFEOpFromWeakForm{T} <: ParamFEOperator{T} end

Most standard instance of ParamFEOperator{T}

"""
struct ParamFEOpFromWeakForm{T} <: ParamFEOperator{T}
  res::Function
  jac::Function
  induced_norm::Function
  pspace::ParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
end

function ParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,pspace,trial,test,args...)

  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq}(
    res,jac,induced_norm,pspace,assem,index_map,trial,test,args...)
end

function LinearParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,pspace,trial,test,args...)

  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  ParamFEOpFromWeakForm{LinearParamEq}(
    res,jac,induced_norm,pspace,assem,index_map,trial,test,args...)
end

FESpaces.get_test(op::ParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::ParamFEOpFromWeakForm) = op.trial
ParamDataStructures.realization(op::ParamFEOpFromWeakForm;kwargs...) = realization(op.pspace;kwargs...)
get_induced_norm(op::ParamFEOpFromWeakForm) = op.induced_norm
ODEs.get_res(op::ParamFEOpFromWeakForm) = op.res
get_jac(op::ParamFEOpFromWeakForm) = op.jac
ODEs.get_assembler(op::ParamFEOpFromWeakForm) = op.assem
IndexMaps.get_index_map(op::ParamFEOpFromWeakForm) = op.index_map

"""
    assemble_norm_matrix(op::ParamFEOperator) -> AbstractMatrix

Assembles the symmetric, positive definite matrix representing a norm operator on
the couple of FESpaces (trial,test) defined in `op`

"""
function assemble_norm_matrix(op::ParamFEOpFromWeakForm)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  inorm = get_induced_norm(op)
  assemble_norm_matrix(inorm,trial,test)
end

function assemble_norm_matrix(f,U::FESpace,V::FESpace)
  assemble_matrix(f,U,V)
end

function assemble_norm_matrix(f,U::TProductFESpace,V::TProductFESpace)
  a = SparseMatrixAssembler(U,V)
  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(U)
  assemble_matrix(a,collect_cell_matrix(U,V,f(u,v)))
end

function assemble_norm_matrix(f,U::TrialFESpace{<:TProductFESpace},V::TProductFESpace)
  assemble_norm_matrix(f,U.space,V)
end

function ParamFEOpFromWeakForm(
  res::Function,
  jac::Function,
  induced_norm::Function,
  pspace::ParamSpace,
  assem::Assembler,
  index_map::FEOperatorIndexMap,
  trial::FESpace,
  test::FESpace,
  coupling::Function)

  op = ParamFEOpFromWeakForm(res,jac,induced_norm,pspace,assem,index_map,trial,test)
  saddlep_op = ParamSaddlePointFEOp(op,coupling)
  return saddlep_op
end

"""
    struct ParamSaddlePointFEOp{T} <: ParamFEOperator{T} end

Interface to deal with the Inf-Sup stability condition of saddle point problems;
the field `coupling` encodes the Inf-Sup operator

"""
struct ParamSaddlePointFEOp{T} <: ParamFEOperator{T}
  op::ParamFEOperator
  coupling::Function
end

FESpaces.get_test(op::ParamSaddlePointFEOp) = get_test(op)
FESpaces.get_trial(op::ParamSaddlePointFEOp) = get_trial(op)
ParamDataStructures.realization(op::ParamSaddlePointFEOp;kwargs...) = realization(op.op;kwargs...)
get_induced_norm(op::ParamSaddlePointFEOp) = get_induced_norm(op)
ODEs.get_res(op::ParamSaddlePointFEOp) = get_res(op.op)
get_jac(op::ParamSaddlePointFEOp) = get_jac(op.op)
ODEs.get_assembler(op::ParamSaddlePointFEOp) = get_assembler(op.op)
IndexMaps.get_index_map(op::ParamSaddlePointFEOp) = get_index_map(op)

"""
    abstract type ParamFEOperatorWithTrian{T} <: ParamFEOperator{T} end

Interface to accommodate the separation of terms in the problem's weak formulation
depending on the triangulation on which the integration occurs. When employing
a ParamFEOperatorWithTrian, the residual and jacobian are returned as [`Contribution`](@ref)
objects, instead of standard arrays. To correctly define an instance of
ParamFEOperatorWithTrian, one needs to:
- provide the integration domains of the residual and jacobian, i.e. their
  respective triangulations
- define the residual and jacobian as functions of the Measure objects corresponding
  to the aforementioned triangulations

Subtypes:

- [`ParamFEOpFromWeakFormWithTrian`](@ref)
- [`ParamSaddlePointFEOpWithTrian`](@ref)
- [`LinearNonlinearParamFEOperatorWithTrian`](@ref)

"""
abstract type ParamFEOperatorWithTrian{T} <: ParamFEOperator{T} end

function FESpaces.get_algebraic_operator(op::ParamFEOperatorWithTrian)
  ParamOpFromFEOpWithTrian(op)
end

"""
    struct ParamFEOpFromWeakFormWithTrian{T} <: ParamFEOperatorWithTrian{T} end

Corresponds to a [`ParamFEOpFromWeakForm`](@ref) object, but in a triangulation
separation setting

"""
struct ParamFEOpFromWeakFormWithTrian{T} <: ParamFEOperatorWithTrian{T}
  op::ParamFEOperator{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jac::Tuple{Vararg{Triangulation}}

  function ParamFEOpFromWeakFormWithTrian(
    op::ParamFEOperator{T},
    trian_res::Tuple{Vararg{Triangulation}},
    trian_jac::Tuple{Vararg{Triangulation}}
    ) where T

    newop = set_triangulation(op,trian_res,trian_jac)
    new{T}(newop,trian_res,trian_jac)
  end
end

function ParamFEOpFromWeakForm(
  res::Function,
  jac::Function,
  induced_norm::Function,
  pspace::ParamSpace,
  index_map::FEOperatorIndexMap,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  trian_res,
  trian_jac)

  op = ParamFEOpFromWeakForm(res,jac,induced_norm,pspace,assem,index_map,trial,test)
  op_trian = ParamFEOpFromWeakFormWithTrian(op,trian_res,trian_jac)
  return op_trian
end

FESpaces.get_test(op::ParamFEOpFromWeakFormWithTrian) = get_test(op.op)
FESpaces.get_trial(op::ParamFEOpFromWeakFormWithTrian) = get_trial(op.op)
ParamDataStructures.realization(op::ParamFEOpFromWeakFormWithTrian;kwargs...) = realization(op.op;kwargs...)
get_induced_norm(op::ParamFEOpFromWeakFormWithTrian) = get_induced_norm(op.op)
ODEs.get_res(op::ParamFEOpFromWeakFormWithTrian) = get_res(op.op)
get_jac(op::ParamFEOpFromWeakFormWithTrian) = get_jac(op.op)
ODEs.get_assembler(op::ParamFEOpFromWeakFormWithTrian) = get_assembler(op.op)
IndexMaps.get_index_map(op::ParamFEOpFromWeakFormWithTrian) = get_index_map(op)

"""
    struct ParamSaddlePointFEOpWithTrian{T} <: ParamFEOperatorWithTrian{T} end

Corresponds to a [`ParamSaddlePointFEOp`](@ref) object, but in a triangulation
separation setting

"""
struct ParamSaddlePointFEOpWithTrian{T} <: ParamFEOperatorWithTrian{T}
  op::ParamSaddlePointFEOp{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jac::Tuple{Vararg{Triangulation}}

  function ParamSaddlePointFEOpWithTrian(
    op::ParamSaddlePointFEOp{T},
    trian_res::Tuple{Vararg{Triangulation}},
    trian_jac::Tuple{Vararg{Triangulation}}) where T

    newop = set_triangulation(op,trian_res,trian_jac)
    new{T}(newop,trian_res,trian_jac)
  end
end

function ParamFEOpFromWeakForm(
  res::Function,
  jac::Function,
  induced_norm::Function,
  pspace::ParamSpace,
  assem::Assembler,
  index_map::FEOperatorIndexMap,
  trial::FESpace,
  test::FESpace,
  coupling::Function,
  trian_res,
  trian_jac)

  saddlep_op = ParamFEOpFromWeakForm(res,jac,induced_norm,pspace,assem,
    index_map,trial,test,coupling)
  saddlep_op_trian = ParamSaddlePointFEOpWithTrian(saddlep_op,trian_res,trian_jac)
  return saddlep_op_trian
end

FESpaces.get_test(op::ParamSaddlePointFEOpWithTrian) = get_test(op.op)
FESpaces.get_trial(op::ParamSaddlePointFEOpWithTrian) = get_trial(op.op)
ParamDataStructures.realization(op::ParamSaddlePointFEOpWithTrian;kwargs...) = realization(op.op;kwargs...)
get_induced_norm(op::ParamSaddlePointFEOpWithTrian) = get_induced_norm(op.op)
ODEs.get_res(op::ParamSaddlePointFEOpWithTrian) = get_res(op.op)
get_jac(op::ParamSaddlePointFEOpWithTrian) = get_jac(op.op)
ODEs.get_assembler(op::ParamSaddlePointFEOpWithTrian) = get_assembler(op.op)
IndexMaps.get_index_map(op::ParamSaddlePointFEOpWithTrian) = get_index_map(op.op)
get_coupling(op::ParamSaddlePointFEOpWithTrian) = get_coupling(op.op)

# utils

function _set_triangulation_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newjac(μ,u,du,v,args...) = jac(μ,u,du,v,args...)
  newjac(μ,u,du,v) = newjac(μ,u,du,v,meas...)
  return newjac
end

function _set_triangulation_res(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newres(μ,u,v,args...) = res(μ,u,v,args...)
  newres(μ,u,v) = newres(μ,u,v,meas...)
  return newres
end

function set_triangulation(op::ParamFEOpFromWeakForm,trian_res,trian_jac)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newjac = _set_triangulation_jac(op.jac,trian_jac,polyn_order)
  ParamFEOpFromWeakForm(
    newres,newjac,op.induced_norm,op.pspace,op.assem,op.index_map,op.trial,op.test)
end

function set_triangulation(op::ParamSaddlePointFEOp,trian_res,trian_jac)
  newop = set_triangulation(op.op,trian_res,trian_jac)
  ParamSaddlePointFEOp(newop,op.coupling)
end

"""
    set_triangulation(op::ParamFEOperatorWithTrian,trian_res,trian_jac) -> ParamFEOperator

Two tuples of triangulations `trian_res` and `trian_jac` are substituted,
respectively, in the residual and jacobian of a ParamFEOperatorWithTrian, and
the resulting ParamFEOperator is returned

"""

function set_triangulation(
  op::ParamFEOperatorWithTrian,
  trian_res=op.trian_res,
  trian_jac=op.trian_jac)

  set_triangulation(op.op,trian_res,trian_jac)
end

"""
    change_triangulation(op::ParamFEOperatorWithTrian,trian_res,trian_jac;kwargs...) -> ParamFEOperatorWithTrian

Replaces the old triangulations relative to the residual and jacobian in `op` with
two new tuples `trian_res` and `trian_jac`, and returns the resulting ParamFEOperatorWithTrian

"""
function change_triangulation(op::ParamFEOperatorWithTrian,trian_res,trian_jac;kwargs...)
  newtrian_res = order_triangulations(op.trian_res,trian_res;kwargs...)
  newtrian_jac = order_triangulations(op.trian_jac,trian_jac;kwargs...)
  newop = set_triangulation(op,newtrian_res,newtrian_jac)
  ParamFEOpFromWeakFormWithTrian(newop,newtrian_res,newtrian_jac)
end

function change_triangulation(op::ParamSaddlePointFEOpWithTrian,trian_res,trian_jac;kwargs...)
  newtrian_res = order_triangulations(op.trian_res,trian_res;kwargs...)
  newtrian_jac = order_triangulations(op.trian_jac,trian_jac;kwargs...)
  newop = set_triangulation(op,newtrian_res,newtrian_jac)
  ParamSaddlePointFEOpWithTrian(newop,newtrian_res,newtrian_jac)
end
