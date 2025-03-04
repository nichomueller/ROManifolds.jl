"""
    struct ParamOpFromFEOp{O,T} <: ParamOperator{O,T}
      op::ParamFEOperator{O,T}
    end

Wrapper that transforms a `ParamFEOperator` into an `ParamOperator`
"""
struct ParamOpFromFEOp{O,T} <: ParamOperator{O,T}
  op::ParamFEOperator{O,T}
end

get_fe_operator(op::ParamOpFromFEOp) = op.op

for f in (:set_domains,:change_domains)
  @eval begin
    function $f(odeop::ParamOpFromFEOp,args...)
      ParamOpFromFEOp($f(odeop.op,args...))
    end
  end
end

"""
    const JointParamOpFromFEOp{O} = ParamOpFromFEOp{O,JointDomains}
"""
const JointParamOpFromFEOp{O} = ParamOpFromFEOp{O,JointDomains}

function Algebra.allocate_residual(
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  res = get_res(op.op)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  b = allocate_vector(assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  res = get_res(op.op)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  matdata = collect_cell_matrix(trial,test,jac(μ,uh,du,v))
  A = allocate_matrix(assem,matdata)

  A
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix_add!(A,assem,matdata)

  A
end

function allocate_lazy_residual(
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_assembler(op.op)

  res = get_res(op.op)
  dc = res(μ,uh,v)
  vecdata,veccache = collect_lazy_cell_vector(test,dc,paramcache.index)
  veccache =
  fill_vecdata!(paramcache,vecdata)
  fill_veccache!(paramcache,veccache)
  b = allocate_vector(assem,vecdata)

  b
end

function lazy_residual!(
  b::AbstractVector,
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache::LazyParamOpCache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  if isstored_vecdata(paramcache)
    vecdata,veccache = get_vec_data_cache(paramcache)
  else
    reset_index!(paramcache)
    uh = EvaluationFunction(paramcache.trial,u)
    test = get_test(op.op)
    v = get_fe_basis(test)
    assem = get_param_assembler(op.op,μ)

    res = get_res(op.op)
    dc = res(μ,uh,v)
    vecdata = collect_lazy_cell_vector(test,dc,paramcache.index)
    fill_vecdata!(paramcache,vecdata)
    _,veccache = get_vec_data_cache(paramcache)
  end

  assemble_lazy_vector_add!(b,assem,(vecdata,veccache,paramcache.index))
  next_index!(paramcache)

  b
end

function allocate_lazy_jacobian(
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  matdata,matcache = collect_lazy_cell_matrix(trial,test,dc,paramcache.index)
  fill_matdata!(paramcache,matdata)
  fill_matcache!(paramcache,matcache)
  A = allocate_matrix(assem,matdata)

  A
end

function lazy_jacobian_add!(
  A::AbstractMatrix,
  op::JointParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix_add!(A,assem,matdata)

  A
end

"""
    const SplitParamOpFromFEOp{O} = ParamOpFromFEOp{O,SplitDomains}
"""
const SplitParamOpFromFEOp{O} = ParamOpFromFEOp{O,SplitDomains}

function Algebra.allocate_residual(
  op::SplitParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  trian_res = get_domains_res(op.op)
  res = get_res(op.op)
  dc = res(μ,uh,v)
  contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
end

function Algebra.residual!(
  b::Contribution,
  op::SplitParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  trian_res = get_domains_res(op.op)
  res = get_res(op.op)
  dc = res(μ,uh,v)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end

  b
end

function Algebra.allocate_jacobian(
  op::SplitParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  trian_jac = get_domains_jac(op.op)
  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  contribution(trian_jac) do trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    allocate_matrix(assem,matdata)
  end
end

function ODEs.jacobian_add!(
  A::Contribution,
  op::SplitParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  trian_jac = get_domains_jac(op.op)
  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  map(A.values,trian_jac) do values,trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix_add!(values,assem,matdata)
  end

  A
end

struct LinearNonlinearParamOpFromFEOp{T} <: ParamOperator{LinearNonlinearParamEq,T}
  op::LinearNonlinearParamFEOperator{T}
end

get_fe_operator(op::LinearNonlinearParamOpFromFEOp) = op.op

function get_linear_operator(op::LinearNonlinearParamOpFromFEOp)
  get_algebraic_operator(get_linear_operator(op.op))
end

function get_nonlinear_operator(op::LinearNonlinearParamOpFromFEOp)
  get_algebraic_operator(get_nonlinear_operator(op.op))
end

function allocate_paramcache(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector)

  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)

  paramcache = allocate_paramcache(op_nlin,μ,u)
  A_lin,b_lin = allocate_systemcache(op_lin,μ,u,paramcache)

  return ParamOpSysCache(paramcache,A_lin,b_lin)
end

function update_paramcache!(
  cache,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization)

  update_paramcache!(cache.paramcache,get_nonlinear_operator(op),μ)
end

function Algebra.allocate_residual(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  b_lin = cache.b
  copy(b_lin)
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A
  copy(A_lin)
end

function Algebra.residual!(
  b,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A
  b_lin = cache.b
  paramcache = cache.paramcache
  residual!(b,get_nonlinear_operator(op),μ,u,paramcache)
  mul!(b,A_lin,u,1,1)
  axpy!(1,b_lin,b)
  b
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A
  paramcache = cache.paramcache
  jacobian_add!(A,get_nonlinear_operator(op),μ,u,paramcache)
  axpy!(1,A_lin,A)
  A
end

# utils

"""
    function collect_lazy_cell_matrix(
      trial::FESpace,
      test::FESpace,
      a::DomainContribution,
      index::Int
      ) -> Tuple{Vector{<:Any},Vector{<:Any},Vector{<:Any}}

For parametric applications, returns the cell-wise data needed to assemble a
global sparse matrix corresponding to the parameter #`index`
"""
function collect_lazy_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  index::Int)

  w = []
  r = []
  c = []
  for strian in get_domains(a)
    scell_mat = get_contribution(a,strian)
    cell_mat, trian = move_contributions(scell_mat,strian)
    @assert ndims(eltype(cell_mat)) == 2
    cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
    cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
    cell_mat_rc_i = lazy_param_getindex(cell_mat_rc,index)
    rows = get_cell_dof_ids(test,trian)
    cols = get_cell_dof_ids(trial,trian)
    push!(w,cell_mat_rc_i)
    push!(r,rows)
    push!(c,cols)
  end
  (w,r,c)
end

"""
    function collect_lazy_cell_vector(
      test::FESpace,
      a::DomainContribution,
      index::Int
      ) -> Tuple{Vector{<:Any},Vector{<:Any}}

For parametric applications, returns the cell-wise data needed to assemble a
global vector corresponding to the parameter #`index`
"""
function collect_lazy_cell_vector(
  test::FESpace,
  a::DomainContribution,
  index::Int)

  w = []
  r = []
  for strian in get_domains(a)
    scell_vec = get_contribution(a,strian)
    cell_vec, trian = move_contributions(scell_vec,strian)
    @assert ndims(eltype(cell_vec)) == 1
    cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
    cell_vec_r_i = lazy_param_getindex(cell_vec_r,index)
    rows = get_cell_dof_ids(test,trian)
    push!(w,cell_vec_r_i)
    push!(r,rows)
  end
  (w,r)
end

"""
    function collect_cell_matrix_for_trian(
      trial::FESpace,
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global sparse matrix for a given
input triangulation `strian`
"""
function collect_cell_matrix_for_trian(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  [cell_mat_rc],[rows],[cols]
end

"""
    function collect_cell_vector_for_trian(
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global vector for a given
input triangulation `strian`
"""
function collect_cell_vector_for_trian(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  [cell_vec_r],[rows]
end
