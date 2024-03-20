abstract type ODEParamOperatorType <: ODEOperatorType end
struct NonlinearParamODE <: ODEParamOperatorType end
struct QuasilinearParamODE <: ODEParamOperatorType end
struct SemilinearParamODE <: ODEParamOperatorType end
struct LinearParamODE <: ODEParamOperatorType end

abstract type ODEParamOperator{T<:ODEParamOperatorType} <: ODEOperator{T} end

function ODEs.allocate_odeopcache(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  args...)

  nothing
end

function ODEs.update_odeopcache!(
  odeopcache,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  args...)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  @abstractmethod
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  b = allocate_residual(odeop,r,us,odeopcache)
  residual!(b,odeop,r,us,odeopcache)
  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOperator,
  r::AbstractTransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  @abstractmethod
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,odeop,r,us,ws,odeopcache)
  A
end

function Algebra.jacobian(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  A = allocate_jacobian(odeop,r,us,odeopcache)
  jacobian!(A,odeop,r,us,ws,odeopcache)
  A
end

mutable struct ParamODEOpFromTFEOpCache <: GridapType
  Us
  Uts
  tfeopcache
  const_forms
end

struct ODEParamOpFromTFEOp{T} <: ODEParamOperator{T}
  op::TransientParamFEOperator{T}
end

Polynomials.get_order(odeop::ODEParamOpFromTFEOp) = get_order(odeop.op)
FESpaces.get_test(odeop::ODEParamOpFromTFEOp) = get_test(odeop.op)
FESpaces.get_trial(odeop::ODEParamOpFromTFEOp) = get_trial(odeop.op)
realization(odeop::ODEParamOpFromTFEOp;kwargs...) = realization(odeop.op;kwargs...)
get_fe_operator(odeop::ODEParamOpFromTFEOp) = odeop.op
ODEs.get_num_forms(odeop::ODEParamOpFromTFEOp) = get_num_forms(odeop.op)
ODEs.get_forms(odeop::ODEParamOpFromTFEOp) = get_forms(odeop.op)
ODEs.is_form_constant(odeop::ODEParamOpFromTFEOp,k::Integer) = is_form_constant(odeop.op,k)

function get_linear_operator(odeop::ODEParamOpFromTFEOp)
  ODEParamOpFromTFEOp(get_linear_operator(odeop.op))
end

function get_nonlinear_operator(odeop::ODEParamOpFromTFEOp)
  ODEParamOpFromTFEOp(get_nonlinear_operator(odeop.op))
end

function ODEs.allocate_odeopcache(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  order = get_order(odeop)
  Ut = get_trial(odeop.op)
  U = allocate_space(Ut)
  Uts = (Ut,)
  Us = (U,)
  for k in 1:order
    Uts = (Uts...,∂t(Uts[k]))
    Us = (Us...,allocate_space(Uts[k+1]))
  end

  tfeopcache = allocate_tfeopcache(odeop.op,r,us)

  uh = ODEs._make_uh_from_us(odeop,us,Us)
  V = get_test(odeop.op)
  v = get_fe_basis(V)
  Ut = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(Ut)
  assem = get_assembler(odeop.op,r)

  const_forms = ()
  num_forms = get_num_forms(odeop.op)
  jacs = get_jacs(odeop.op)

  μ,t = get_params(r),get_times(r)

  dc = DomainContribution()
  for k in 0:order
    jac = jacs[k+1]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(Ut,V,dc)
  A_full = allocate_matrix(assem,matdata)

  for k in 0:num_forms-1
    const_form = nothing
    if is_form_constant(odeop,k)
      jac = jacs[k+1]
      dc = jac(μ,t,uh,du,v)
      matdata = collect_cell_matrix(Ut,V,dc)
      const_form = copy(A_full)
      fillstored!(const_form,zero(eltype(const_form)))
      assemble_matrix_add!(const_form,assem,matdata)
    end
    const_forms = (const_forms...,const_form)
  end

  ODEOpFromTFEOpCache(Us,Uts,tfeopcache,const_forms)
end

function ODEs.update_odeopcache!(odeopcache,odeop::ODEParamOpFromTFEOp,r::TransientParamRealization)
  Us = ()
  for k in 0:get_order(odeop)
    Us = (Us...,evaluate!(odeopcache.Us[k+1],odeopcache.Uts[k+1],r))
  end
  odeopcache.Us = Us

  tfeopcache,op = odeopcache.tfeopcache,odeop.op
  odeopcache.tfeopcache = update_tfeopcache!(tfeopcache,op,r)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.op)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  vecdata = collect_cell_vector(V,res(μ,t,uh,v))
  allocate_vector(assem,vecdata)
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.op)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)
  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp{SemilinearParamODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = RB._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.tfeop)
  dc = res(μ,t,uh,v)

  order = get_order(odeop)
  mass = get_forms(odeop.tfeop)[1]
  ∂tNuh = ∂t(uh,Val(order))
  dc = dc + mass(μ,t,∂tNuh,v)

  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,assem,vecdata)

  r
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp{LinearParamODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.op)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  # Forms
  order = get_order(odeop)
  forms = get_forms(odeop.op)
  ∂tkuh = uh
  for k in 0:order
    form = forms[k+1]
    dc = dc + form(μ,t,∂tkuh,v)
    if k < order
      ∂tkuh = ∂t(∂tkuh)
    end
  end

  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  Ut = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.op)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 0:get_order(odeop.op)
    jac = jacs[k+1]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(Ut,V,dc)
  allocate_matrix(assem,matdata)
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  Ut = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.op)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 0:get_order(odeop)
    w = ws[k+1]
    iszero(w) && continue
    jac = jacs[k+1]
    dc = dc + w * jac(μ,t,uh,du,v)
  end

  if num_domains(dc) > 0
    matdata = collect_cell_matrix(Ut,V,dc)
    assemble_matrix_add!(A,assem,matdata)
  end

  A
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOpFromTFEOp{LinearParamODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  Ut = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.op)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 0:get_order(odeop)
    w = ws[k+1]
    iszero(w) && continue
    if is_form_constant(odeop,k)
      axpy_entries!(w,odeopcache.const_forms[k+1],A)
    else
      jac = jacs[k+1]
      dc = dc + w * jac(μ,t,uh,du,v)
    end
  end

  if num_domains(dc) > 0
    matdata = collect_cell_matrix(Ut,V,dc)
    assemble_matrix_add!(A,assem,matdata)
  end

  A
end
