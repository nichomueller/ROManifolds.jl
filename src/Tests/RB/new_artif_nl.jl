begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

struct TempPTFEOperator <: PTFEOperator{Nonlinear}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  pspace::PSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
  function TempPTFEOperator(
    res::Function,jac::Function,jac_t::Function,pspace,trial,test)
    assem = SparseMatrixAssembler(trial,test)
    new(res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
  end
end

function Base.getindex(op::TempPTFEOperator,row,col)
  if isa(get_test(op),MultiFieldFESpace)
    trials_col = get_trial(op)[col]
    test_row = op.test[row]
    sf(q,idx) = single_field(op,q,idx)
    res(μ,t,u,dv) = op.res(μ,t,sf(u,col),sf(dv,row))
    jac(μ,t,u,du,dv) = op.jacs[1](μ,t,sf(u,col),sf(du,col),sf(dv,row))
    jac_t(μ,t,u,dut,dv) = op.jacs[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    return TempPTFEOperator(res,jac,jac_t,op.pspace,trials_col,test_row)
  else
    return op
  end
end

struct TempPODEOperator <: PODEOperator{Nonlinear}
  feop::TempPTFEOperator
end

function TransientFETools.get_algebraic_operator(feop::TempPTFEOperator)
  TempPODEOperator(feop)
end

struct TempPTAlgebraicOperator <: PTAlgebraicOperator{Nonlinear}
  odeop::PODEOperator
  μ
  tθ
  dtθ::Float
  u0::PTArray
  ode_cache
  vθ::PTArray
end

function get_ptoperator(
  odeop::TempPODEOperator,μ,tθ,dtθ::Float,u0::PTArray,ode_cache,vθ::PTArray)
  TempPTAlgebraicOperator(odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function solve_step!(
  uf::PTArray,
  solver::PThetaMethod,
  op::PODEOperator,
  μ::AbstractVector,
  u0::PTArray,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op,μ,tθ)
    vθ = similar(u0)
    vθ .= 0.0
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  nlop = TempPTAlgebraicOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

function residual!(
  b::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

function residual_for_trian!(
  b::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual_for_trian!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

function residual_for_idx!(
  b::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

function jacobian!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache,args...)
end

function jacobian!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache,args...)
end

function jacobian_for_trian!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian_for_trian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache,args...)
end

function jacobian_for_idx!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache,args...)
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace,
  μ::Table)

  nsnaps_mdeim = info.nsnaps_mdeim
  _μ = μ[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,_μ)
  rhs = collect_compress_rhs(info,op,rbspace)
  show(rhs)
  rhs
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace,
  μ::Table)

  nsnaps_mdeim = info.nsnaps_mdeim
  θ = fesolver.θ
  _μ = μ[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,_μ)
  lhs = collect_compress_lhs(info,op,rbspace;θ)
  show(lhs)
  lhs
end

Base.:(∘)(::Function,::Tuple{Vararg{Union{Nothing,CellField}}}) = nothing

begin
  mesh = "model_circle_2D_coarse.json"
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  order = 2
  degree = 4

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)
  g0(x,μ,t) = VectorValue(0,0)
  g0(μ,t) = x->g0(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
  dc(u,du,v) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)

  res(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) + c(u,v) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) + dc(u,du,v) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  trial_u = PTTrialFESpace(test_u,[g0,g])
  test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = TempPTFEOperator(res,jac,jac_t,pspace,trial,test)

  t0,tf,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))

  nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
  fesolver = PThetaMethod(nls,xh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = false
  save_structures = true
  norm_style = [:l2,:l2]
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_mdeim = 30
  nsnaps_test = 10
  st_mdeim = false
  postprocess = true
  info = RBInfo(test_path;ϵ,norm_style,compute_supremizers,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim,postprocess)
end

sols,params = load(info,(BlockSnapshots,Table))
rbspace = load(info,BlockRBSpace)
rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)

snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
times = get_times(fesolver)
xn,μn = PTArray(snaps_test[1:length(times)]),params_test[1]
x = zero(xn)
op = get_ptoperator(fesolver,feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
xrb = space_time_projection(x,op,rbspace)
for iter in 1:20
  rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
  lhs,lhs_t = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
  _lhs = lhs + lhs_t
  _rhs = rhs + lhs_t*xrb
  dxrb = PTArray(_lhs[1] \ _rhs[1])
  xrb -= dxrb
  x = recast(xrb,rbspace)
  op = update_ptoperator(op,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))

################################################################################
function Base.zero(f::TransientCellField)
  zero_values = get_free_dof_values(f.cellfield)
  cellfield = EvaluationFunction(f.cellfield.fe_space,zero_values)
  derivatives = ()
  for i = eachindex(f.derivatives)
    derivatives = (derivatives...,EvaluationFunction(f.derivatives[i].fe_space,zero_values))
  end
  TransientCellField(cellfield,derivatives)
end

# nonlinear part
_c(u,du,v) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
dir(μ,t) = zero(trial_u(μ,t))
res_nlin(μ,t,(u0,p0),(u,p),(v,q)) = (∫ₚ(v⋅∂ₚt(u0),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u0),dΩ) + _c(u,u0,v)
  - ∫ₚ(p0*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u0)),dΩ))
res_nlin(μ,t,u,v) = res_nlin(μ,t,zero(u),u,v)
jac_nlin(μ,t,(u,p),(du,dp),(v,q)) = (∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) + dc(u,du,v)
  - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ))
jac_t_nlin(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
feop_nlin = PTFEOperator(res_nlin,jac_nlin,jac_t_nlin,pspace,trial,test)
rbrhs_nlin,rblhs_nlin = collect_compress_rhs_lhs(info,feop_nlin,fesolver,rbspace,params)

# aux part
res_aux(μ,t,(u,p),(v,q)) = nothing
jac_aux(μ,t,(u,p),(du,dp),(v,q)) = (∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) + _c(u,du,v)
  - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ))
jac_t_aux(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
feop_aux = PTFEOperator(res_aux,jac_aux,jac_t_aux,pspace,trial,test)
rblhs_aux = collect_compress_lhs(info,feop_aux,fesolver,rbspace,params)

x = copy(xn) .* 0.
op_nlin = get_ptoperator(fesolver,feop_nlin,x,Table([μn]))
op_aux = get_ptoperator(fesolver,feop_aux,x,Table([μn]))
xrb = space_time_projection(x,op_nlin,rbspace)
rhs_cache,lhs_cache = allocate_cache(op_nlin,x)

for iter in 1:20
  rhs_nlin = collect_rhs_contributions!(rhs_cache,info,op_nlin,rbrhs_nlin,rbspace)
  lhs_nlin,lhs_t = collect_lhs_contributions!(lhs_cache,info,op_nlin,rblhs_nlin,rbspace)
  lhs_aux, = collect_lhs_contributions!(lhs_cache,info,op_aux,rblhs_aux,rbspace)
  lhs = lhs_t+lhs_nlin
  rhs = rhs_nlin+(lhs_t+lhs_aux)*xrb
  dxrb = PTArray(lhs[1] \ rhs[1])
  xrb -= dxrb
  x = recast(xrb,rbspace)
  op_nlin = update_ptoperator(op_nlin,x)
  op_aux = update_ptoperator(op_aux,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))
