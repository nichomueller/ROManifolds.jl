"""
A parametric version of the `Gridap` `TransientFEOperator`
"""

abstract type TransientParamFEOperator{T<:ODEParamOperatorType} <: TransientFEOperator{T} end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperator)
  ODEParamOpFromTFEOp(feop)
end

function ODEs.allocate_tfeopcache(
  feop::TransientParamFEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  nothing
end

function ODEs.update_tfeopcache!(
  tfeopcache,
  feop::TransientParamFEOperator,
  r::TransientParamRealization)

  tfeopcache
end

realization(op::TransientParamFEOperator;kwargs...) = @abstractmethod
get_induced_norm(op::TransientParamFEOperator) = @abstractmethod

function assemble_norm_matrix(op::TransientParamFEOperator)
  @abstractmethod
end

get_coupling(op::TransientParamFEOperator) = @abstractmethod

IndexMaps.get_index_map(op::TransientParamFEOperator) = @abstractmethod

function assemble_coupling_matrix(op::TransientParamFEOperator)
  @abstractmethod
end

get_linear_operator(op::TransientParamFEOperator) = @abstractmethod
get_nonlinear_operator(op::TransientParamFEOperator) = @abstractmethod

struct TransientParamFEOpFromWeakForm <: TransientParamFEOperator{NonlinearParamODE}
  res::Function
  jacs::Tuple{Vararg{Function}}
  induced_norm::Function
  tpspace::TransientParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
  order::Integer
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},induced_norm::Function,tpspace,trial,test,args...)

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  TransientParamFEOpFromWeakForm(
    res,jacs,induced_norm,tpspace,assem,index_map,trial,test,order,args...)
end

function TransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test,args...)

  TransientParamFEOperator(res,(jac,),induced_norm,tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test,args...)

  TransientParamFEOperator(res,(jac,jac_t),induced_norm,tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,induced_norm::Function,tpspace,trial,test,args...;order::Integer=1)

  function jac_0(μ,t,u,du,v,args...)
    function res_0(y)
      u0 = TransientCellField(y,u.derivatives)
      res(μ,t,u0,v,args...)
    end
    jacobian(res_0,u.cellfield)
  end
  jacs = (jac_0,)

  for k in 1:order
    function jac_k(μ,t,u,duk,v,args...)
      function res_k(y)
        derivatives = (u.derivatives[1:k-1]...,y,u.derivatives[k+1:end]...)
        uk = TransientCellField(u.cellfield,derivatives)
        res(μ,t,uk,v,args...)
      end
      jacobian(res_k,u.derivatives[k])
    end
    jacs = (jacs...,jac_k)
  end

  TransientParamFEOperator(res,jacs,induced_norm,tpspace,trial,test,args...)
end

FESpaces.get_test(op::TransientParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamFEOpFromWeakForm) = op.trial
ReferenceFEs.get_order(op::TransientParamFEOpFromWeakForm) = op.order
ODEs.get_res(op::TransientParamFEOpFromWeakForm) = op.res
ODEs.get_jacs(op::TransientParamFEOpFromWeakForm) = op.jacs
ODEs.get_assembler(op::TransientParamFEOpFromWeakForm) = op.assem
IndexMaps.get_index_map(op::TransientParamFEOpFromWeakForm) = op.index_map
realization(op::TransientParamFEOpFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)
get_induced_norm(op::TransientParamFEOpFromWeakForm) = op.induced_norm

function assemble_norm_matrix(op::TransientParamFEOpFromWeakForm)
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

function ODEs.get_assembler(feop::TransientParamFEOpFromWeakForm,r::TransientParamRealization)
  get_param_assembler(get_assembler(feop),r)
end

struct TransientParamSemilinearFEOpFromWeakForm <: TransientParamFEOperator{SemilinearParamODE}
  mass::Function
  res::Function
  jacs::Tuple{Vararg{Function}}
  constant_mass::Bool
  induced_norm::Function
  tpspace::TransientParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
  order::Integer
end

function TransientParamSemilinearFEOperator(
  mass::Function,res::Function,jacs::Tuple{Vararg{Function}},
  induced_norm::Function,tpspace,trial,test,
  args...;constant_mass::Bool=false)

  order = length(jacs)
  jac_N(μ,t,u,du,v,args...) = mass(μ,t,du,v,args...)
  jacs = (jacs...,jac_N)
  if order == 0
    @warn ODEs.default_linear_msg
    return TransientParamLinearFEOperator(
      (mass,),res,jacs,induced_norm,tpspace,trial,test;constant_forms=(constant_mass,))
  end
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  TransientParamSemilinearFEOpFromWeakForm(
    mass,res,jacs,constant_mass,induced_norm,tpspace,assem,index_map,trial,test,order,args...)
end

function TransientParamSemilinearFEOperator(
  mass::Function,res::Function,jac::Function,
  induced_norm::Function,tpspace,trial,test,args...;kwargs...)

  TransientParamSemilinearFEOperator(mass,res,(jac,),induced_norm,
    tpspace,trial,test,args...;kwargs...)
end

function TransientParamSemilinearFEOperator(
  mass::Function,res::Function,jac::Function,jac_t::Function,
  induced_norm::Function,tpspace,trial,test,args...;kwargs...)

  TransientParamSemilinearFEOperator(mass,res,(jac,jac_t),induced_norm,
    tpspace,trial,test,args...;kwargs...)
end

function TransientParamSemilinearFEOperator(
  mass::Function,res::Function,induced_norm::Function,tpspace,trial,test,
  args...;order::Integer=1,kwargs...)

  if order == 0
    @warn ODEs.default_linear_msg
    return TransientLinearFEOperator(mass,res,trial,test;kwargs...)
  end

  jacs = ()
  if order > 0
    function jac_0(μ,t,u,du,v,args...)
      function res_0(y)
        u0 = TransientCellField(y,u.derivatives)
        res(μ,t,u0,v,args...)
      end
      jacobian(res_0,u.cellfield)
    end
    jacs = (jacs...,jac_0)
  end

  for k in 1:order-1
    function jac_k(μ,t,u,duk,v,args...)
      function res_k(y)
        derivatives = (u.derivatives[1:k-1]...,y,u.derivatives[k+1:end]...)
        uk = TransientCellField(u.cellfield,derivatives)
        res(t,uk,v,args...)
      end
      jacobian(res_k,u.derivatives[k])
    end
    jacs = (jacs...,jac_k)
  end

  TransientParamSemilinearFEOperator(mass,res,jacs,induced_norm,tpspace,trial,test,args...;kwargs...)
end

FESpaces.get_test(op::TransientParamSemilinearFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamSemilinearFEOpFromWeakForm) = op.trial
ReferenceFEs.get_order(op::TransientParamSemilinearFEOpFromWeakForm) = op.order
ODEs.get_res(op::TransientParamSemilinearFEOpFromWeakForm) = op.res
ODEs.get_jacs(op::TransientParamSemilinearFEOpFromWeakForm) = op.jacs
ODEs.get_assembler(op::TransientParamSemilinearFEOpFromWeakForm) = op.assem
realization(op::TransientParamSemilinearFEOpFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)
get_induced_norm(op::TransientParamSemilinearFEOpFromWeakForm) = op.induced_norm

function ODEs.is_form_constant(op::TransientParamSemilinearFEOpFromWeakForm,k::Integer)
  (k == get_order(op)+1) && op.constant_mass
end

function assemble_norm_matrix(op::TransientParamSemilinearFEOpFromWeakForm)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  inorm = get_induced_norm(op)
  assemble_norm_matrix(inorm,trial,test)
end

function ODEs.get_assembler(feop::TransientParamSemilinearFEOpFromWeakForm,r::TransientParamRealization)
  get_param_assembler(get_assembler(feop),r)
end

struct TransientParamLinearFEOpFromWeakForm <: TransientParamFEOperator{LinearParamODE}
  forms::Tuple{Vararg{Function}}
  res::Function
  jacs::Tuple{Vararg{Function}}
  constant_forms::Tuple{Vararg{Bool}}
  induced_norm::Function
  tpspace::TransientParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
  order::Integer
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,induced_norm::Function,tpspace,trial,test,
  args...;constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms)))

  order = length(forms)-1
  jacs = ntuple(k -> ((μ,t,u,duk,v,args...) -> forms[k](μ,t,duk,v,args...)),length(forms))
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  TransientParamLinearFEOpFromWeakForm(
    forms,res,jacs,constant_forms,induced_norm,tpspace,assem,index_map,trial,test,order,args...)
end

function TransientParamLinearFEOperator(
  mass::Function,res::Function,induced_norm::Function,tpspace,trial,test,
  args...;kwargs...)

  TransientParamLinearFEOperator((mass,),res,induced_norm,tpspace,trial,test;kwargs)
end

function TransientParamLinearFEOperator(
  stiffness::Function,mass::Function,res::Function,induced_norm::Function,tpspace,trial,test,
  args...;kwargs...)

  TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,tpspace,
    trial,test,args...;kwargs...)
end

function TransientParamLinearFEOperator(
  stiffness::Function,damping::Function,mass::Function,res::Function,
  induced_norm::Function,tpspace,trial,test,args...;kwargs...)

  TransientParamLinearFEOperator((stiffness,damping,mass),res,induced_norm,tpspace,
    trial,test,args...;kwargs...)
end

FESpaces.get_test(op::TransientParamLinearFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamLinearFEOpFromWeakForm) = op.trial
ReferenceFEs.get_order(op::TransientParamLinearFEOpFromWeakForm) = op.order
ODEs.get_res(op::TransientParamLinearFEOpFromWeakForm) = op.res
ODEs.get_jacs(op::TransientParamLinearFEOpFromWeakForm) = op.jacs
ODEs.get_assembler(op::TransientParamLinearFEOpFromWeakForm) = op.assem
realization(op::TransientParamLinearFEOpFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)
get_induced_norm(op::TransientParamLinearFEOpFromWeakForm) = op.induced_norm

function ODEs.is_form_constant(op::TransientParamLinearFEOpFromWeakForm,k::Integer)
  op.constant_forms[k]
end

function assemble_norm_matrix(op::TransientParamLinearFEOpFromWeakForm)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  inorm = get_induced_norm(op)
  assemble_norm_matrix(inorm,trial,test)
end

function ODEs.get_assembler(feop::TransientParamLinearFEOpFromWeakForm,r::TransientParamRealization)
  get_param_assembler(get_assembler(feop),r)
end

function test_transient_fe_operator(op::TransientParamFEOperator,uh,μt)
  odeop = get_algebraic_operator(op)
  @test isa(odeop,ODEParamOperator)
  cache = allocate_cache(op)
  V = get_test(op)
  @test isa(V,FESpace)
  U = get_trial(op)
  U0 = U(μt)
  @test isa(U0,TrialParamFESpace)
  r = allocate_residual(op,μt,uh,cache)
  @test isa(r,ParamVector)
  xh = TransientCellField(uh,(uh,))
  residual!(r,op,μt,xh,cache)
  @test isa(r,ParamVector)
  J = allocate_jacobian(op,μt,uh,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,1,1.0,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,2,1.0,cache)
  @test isa(J,ParamMatrix)
  cache = update_cache!(cache,op,μt)
  true
end
