abstract type ODEParamOperatorType <: ODEOperatorType end

struct NonlinearParamODE <: ODEParamOperatorType end

struct LinearParamODE <: ODEParamOperatorType end

abstract type ODEParamOperator{T<:ODEParamOperatorType} <: ODEOperator{T} end

Polynomials.get_order(op::ODEParamOpFromTFEOp) = get_order(op.feop)
FESpaces.get_test(op::ODEParamOpFromTFEOp) = get_test(op.feop)
FESpaces.get_trial(op::ODEParamOpFromTFEOp) = get_trial(op.feop)
realization(op::ODEParamOpFromTFEOp;kwargs...) = realization(op.feop;kwargs...)
get_fe_operator(op::ODEParamOpFromTFEOp) = op.feop
get_linear_operator(op::ODEParamOpFromTFEOp) = ODEParamOpFromTFEOp(get_linear_operator(op.feop))
get_nonlinear_operator(op::ODEParamOpFromTFEOp) = ODEParamOpFromTFEOp(get_nonlinear_operator(op.feop))

function ODEs.allocate_odeopcache(
  odeop::ODEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  args...)

  nothing
end

function ODEs.update_odeopcache!(
  odeopcache,
  odeop::ODEOperator,
  r::TransientParamRealization,
  args...)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  @abstractmethod
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  odeop::ODEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  b = allocate_residual(odeop,r,us,odeopcache)
  residual!(b,odeop,r,us,odeopcache)
  b
end

function Algebra.allocate_jacobian(
  odeop::ODEOperator,
  r::AbstractTransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  @abstractmethod
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  odeop::ODEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,odeop,r,us,ws,odeopcache)
  A
end

function Algebra.jacobian(
  odeop::ODEOperator,
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
  tfeop::TransientParamFEOperator{T}
end

function Polynomials.get_order(odeop::ODEParamOpFromTFEOp)
  get_order(odeop.tfeop)
end

function ODEs.get_num_forms(odeop::ODEParamOpFromTFEOp)
  get_num_forms(odeop.tfeop)
end

function ODEs.get_forms(odeop::ODEParamOpFromTFEOp)
  get_forms(odeop.tfeop)
end

function ODEs.is_form_constant(odeop::ODEParamOpFromTFEOp,k::Integer)
  is_form_constant(odeop.tfeop,k)
end

function allocate_odeopcache(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  order = get_order(odeop)
  Ut = get_trial(odeop.tfeop)
  U = allocate_space(Ut)
  Uts = (Ut,)
  Us = (U,)
  for k in 1:order
    Uts = (Uts...,∂t(Uts[k]))
    Us = (Us...,allocate_space(Uts[k+1]))
  end

  tfeopcache = allocate_tfeopcache(odeop.tfeop,r,us)

  uh = ODEs._make_uh_from_us(odeop,us,Us)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  assem = get_assembler(odeop.tfeop,r)

  const_forms = ()
  num_forms = get_num_forms(odeop.tfeop)
  jacs = get_jacs(odeop.tfeop)

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

  tfeopcache,tfeop = odeopcache.tfeopcache,odeop.tfeop
  odeopcache.tfeopcache = update_tfeopcache!(tfeopcache,tfeop,r)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.tfeop)
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
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.tfeop)
  dc = res(μ,t,uh,v)
  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp{<:AbstractLinearODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop.tfeop)
  dc = res(μ,t,uh,v)

  # Forms
  order = get_order(odeop)
  forms = get_forms(odeop.tfeop)
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
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.tfeop)
  dc = DomainContribution()
  for k in 0:get_order(odeop.tfeop)
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
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.tfeop)
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
  odeop::ODEParamOpFromTFEOp{<:AbstractLinearODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.tfeop)
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
