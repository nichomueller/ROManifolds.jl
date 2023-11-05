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

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::TempPTFEOperator,
  fesolver::PThetaMethod,
  rbspace,
  snaps,
  μ::Table)

  nsnaps_mdeim = info.nsnaps_mdeim
  θ = fesolver.θ

  snapsθ = recenter(snaps,fesolver.uh0(μ);θ)
  _snapsθ,_μ = snapsθ[1:nsnaps_mdeim],μ[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,_snapsθ,_μ)
  rhs = collect_compress_rhs(info,op,rbspace)
  lhs = collect_compress_lhs(info,op,rbspace;θ)
  show(rhs),show(lhs)

  rhs,lhs
end

Base.:(∘)(::Function,::Tuple{Vararg{Union{Nothing,CellField}}}) = nothing

begin
  mesh = "cube2x2.json"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
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
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial_u = PTTrialFESpace(test_u,g)
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

if false
  times = get_times(fesolver)
  params = realization(feop,nsnaps_state+nsnaps_test)
  sols,stats = collect_multi_field_solutions(fesolver,feop,params)
  rbspace = reduced_basis(info,feop,sols)
  rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)

  save(info,(sols,params,stats))
  save(info,(rbspace,rbrhs,rblhs))
else
  sols,params = load(info,(BlockSnapshots,Table))
  rbspace = load(info,BlockRBSpace)
  rbrhs,rblhs = load(info,(BlockRBVecAlgebraicContribution,Vector{BlockRBMatAlgebraicContribution}))
end

snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
times = get_times(fesolver)
xn,μn = PTArray(vcat(snaps_test...)[1:length(times)]),params_test[1]
op = get_ptoperator(fesolver,feop,xn,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,xn)

nu = get_rb_ndofs(rbspace[1])
np = get_rb_ndofs(rbspace[2])

# THIS WORKS -- GRIDAP OK
################################# GRIDAP #######################################
g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ + ∫(v⊙(∇(u)'⋅du))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,u,v)
trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(res_ok,jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)
Nu,Np = test_u.nfree,length(get_free_dof_ids(test_p))
v0 = zeros(Nu+Np)
nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,v0,ode_cache_ok,v0)
bok = allocate_residual(nlop,v0)
Aok = allocate_jacobian(nlop,v0)

function gridap_quantities(x,k)
  xk = x[k]
  tk = times[k]
  ode_cache = nlop.ode_cache
  ode_op = nlop.odeop
  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,ode_op,tk)

  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  fill!(bok,z)
  residual!(bok,ode_op,tk,(xk,v0),ode_cache)
  jacobians!(Aok,ode_op,tk,(xk,v0),(1.0,1/(dt*θ)),ode_cache)
  return bok,Aok
end

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)
x = copy(xn) .* 0.
op = get_ptoperator(fesolver,feop,x,Table([μn]))
for iter in 1:fesolver.nls.max_nliters
  A,b = [],[]
  for k = eachindex(times)
    bk,Ak = gridap_quantities(x,k)
    push!(b,copy(bk))
    push!(A,copy(Ak))
  end

  xrb = space_time_projection(x,op,rbspace)

  LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/(θ*dt)),A)...)
  LHS11_2 = NnzMatrix([NnzVector(M/(θ*dt)) for _ = times]...)
  LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),A)...)
  LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),A)...)

  LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
  LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
  LHS11_rb = LHS11_rb_1 + LHS11_rb_2
  LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
  LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

  LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

  R1 = NnzMatrix(map(x->x[1:Nu],b)...)
  R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
  RHS1_rb = space_time_projection(R1,rbspace[1])
  RHS2_rb = space_time_projection(R2,rbspace[2])
  RHS_rb = vcat(RHS1_rb + LHS11_rb_2*xrb[1][1:nu],RHS2_rb)

  dxrb = LHS_rb \ RHS_rb
  dxrb_1,dxrb_2 = dxrb[1:nu],dxrb[1+nu:end]
  xiter = vcat(recast(PTArray(dxrb_1),rbspace[1]),recast(PTArray(dxrb_2),rbspace[2]))
  x -= xiter

  nerr = norm(dxrb)
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))
################################# GRIDAP #######################################

# THIS WORKS -- RB JACOBIAN OK
################################# HYBRID #######################################
x = copy(xn) .* 0.
op = get_ptoperator(fesolver,feop,x,Table([μn]))
for iter in 1:fesolver.nls.max_nliters
  A,b = [],[]
  for k = eachindex(times)
    bk,Ak = gridap_quantities(x,k)
    push!(b,copy(bk))
    push!(A,copy(Ak))
  end

  xrb = space_time_projection(x,op,rbspace)

  LHS_rb = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)[1]
  LHS11_2 = NnzMatrix([NnzVector(M/(θ*dt)) for _ = times]...)
  LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)

  R1 = NnzMatrix(map(x->x[1:Nu],b)...)
  R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
  RHS1_rb = space_time_projection(R1,rbspace[1])
  RHS2_rb = space_time_projection(R2,rbspace[2])
  RHS_rb = vcat(RHS1_rb + LHS11_rb_2*xrb[1][1:nu],RHS2_rb)

  dxrb = LHS_rb \ RHS_rb
  dxrb_1,dxrb_2 = dxrb[1:nu],dxrb[1+nu:end]
  xiter = vcat(recast(PTArray(dxrb_1),rbspace[1]),recast(PTArray(dxrb_2),rbspace[2]))
  x -= xiter

  nerr = norm(dxrb)
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))

################################# HYBRID #######################################

# SOMETHING WRONG WITH RESIDUAL
################################## MDEIM #######################################
x = copy(xn) .* 0.
op = get_ptoperator(fesolver,feop,x,Table([μn]))
xrb = space_time_projection(x,op,rbspace)
A,b = [],[]
for k = eachindex(times)
  bk,Ak = gridap_quantities(x,k)
  push!(b,copy(bk))
  push!(A,copy(Ak))
end

LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/(θ*dt)),A)...)
LHS11_2 = NnzMatrix([NnzVector(M/(θ*dt)) for _ = times]...)
LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),A)...)
LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),A)...)

LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
LHS11_rb = LHS11_rb_1 + LHS11_rb_2
LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

R1 = NnzMatrix(map(x->x[1:Nu],b)...)
R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
RHS1_rb = space_time_projection(R1,rbspace[1])
RHS2_rb = space_time_projection(R2,rbspace[2])
RHS_rb = vcat(RHS1_rb,RHS2_rb)
_RHS_rb = vcat(LHS11_rb_2*xrb[1][1:nu],zeros(np)) + RHS_rb

rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
lhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)

norm(lhs[1] - LHS_rb,Inf)
norm(rhs[1] - RHS_rb,Inf)
norm(rhs[1] - _RHS_rb,Inf)

# THIS DOES NOT WORK
################################## MDEIM #######################################
x = copy(xn) .* 0.
op = get_ptoperator(fesolver,feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
for iter in 1:fesolver.nls.max_nliters
  xrb = space_time_projection(x,op,rbspace)
  rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
  dxrb = PTArray(lhs[1] \ (rhs[1] )) # + vcat(LHS11_rb_2*xrb[1][1:nu],zeros(np))
  xrb -= dxrb
  x .+= recast(xrb,rbspace)
  op = update_ptoperator(op,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end
################################## MDEIM #######################################
dir(μ,t) = zero(trial_u(μ,t))
ddir(μ,t) = zero(∂ₚt(trial_u)(μ,t))
_c(u,du,v) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
_res(μ,t,(u,p),(v,q)) = (∫ₚ(v⋅ddir(μ,t),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(dir(μ,t)),dΩ) + _c(u,dir(μ,t),v)
  - ∫ₚ(zero(test_p)*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(dir(μ,t))),dΩ))
_jac(μ,t,(u,p),(du,dp),(v,q)) = (∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) + _c(u,du,v) -
  ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ))
_feop = TempPTFEOperator(_res,_jac,jac_t,pspace,trial,test)

_rbrhs,_rblhs = collect_compress_rhs_lhs(info,_feop,fesolver,rbspace,sols,params)

x = copy(xn) .* 0.
op = get_ptoperator(fesolver,feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
for iter in 1:fesolver.nls.max_nliters
  xrb = space_time_projection(x,op,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
  _lhs = collect_lhs_contributions!(lhs_cache,info,op,_rblhs,rbspace)
  _rhs = collect_rhs_contributions!(rhs_cache,info,op,_rbrhs,rbspace)
  dxrb = NonaffinePTArray([lhs[1] \ (_lhs[1]*xrb[1]+_rhs[1])])
  xrb -= dxrb
  x = recast(xrb,rbspace)
  op = update_ptoperator(op,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

# FROM SCRATCH
res_stokes(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
jac_stokes(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
jac_t_stokes(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
feop_stokes = TempPTFEOperator(res_stokes,jac_stokes,jac_t_stokes,pspace,trial,test)
rbrhs_stokes,rblhs_stokes = collect_compress_rhs_lhs(info,feop_stokes,fesolver,rbspace,sols,params)

nothing_res(μ,t,(u,p),(v,q)) = nothing
nothing_jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
feop_dc = TempPTFEOperator(nothing_res,dc,nothing_jac_t,pspace,trial,test)
rblhs_dc = collect_compress_lhs(info,feop_dc,fesolver,rbspace,sols,params)

_clift(μ,t,(u,p),(v,q)) = ∫ₚ(v⊙(∇(zero(trial_u(μ,t)))'⋅u),dΩ)
_c(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
feop_c = TempPTFEOperator(_clift,_c,nothing_jac_t,pspace,trial,test)
rbrhs_c,rblhs_c = collect_compress_rhs_lhs(info,feop_c,fesolver,rbspace,sols,params)

x = copy(xn) .* 0.
op = get_ptoperator(fesolver,feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
for iter in 1:fesolver.nls.max_nliters
  xrb = space_time_projection(x,op,rbspace)
  lhs_stokes = collect_lhs_contributions!(lhs_cache,info,op,rblhs_stokes,rbspace)
  rhs_stokes = collect_rhs_contributions!(rhs_cache,info,op,rbrhs_stokes,rbspace)
  lhs_dc = collect_lhs_contributions!(lhs_cache,info,op,rblhs_dc,rbspace)
  lhs_c = collect_lhs_contributions!(lhs_cache,info,op,rblhs_c,rbspace)
  rhs_c = collect_rhs_contributions!(rhs_cache,info,op,rbrhs_c,rbspace)

  lhs = lhs_stokes[1] + lhs_dc[1]
  rhs = (lhs_stokes[1]+lhs_c[1])*xrb[1] + rhs_stokes[1] - rhs_c[1]

  dxrb = NonaffinePTArray([lhs \ rhs])
  xrb -= dxrb
  x = recast(xrb,rbspace)
  op = update_ptoperator(op,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

################################ CHECK NL MDEIM ################################
_res(μ,t,(u,p),(v,q)) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
_jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)
_jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
_feop = TempPTFEOperator(_res,_jac,_jac_t,pspace,trial,test)
_rbrhs,_rblhs = collect_compress_rhs_lhs(info,_feop,fesolver,rbspace,sols,params)

g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)

res_ok(t,(u,p),(v,q)) = ∫(v⊙(conv∘(u,∇(u))))dΩ
jac_ok(t,(u,p),(du,dp),(v,q)) =∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ
jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = nothing
trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(res_ok,jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)
Nu,Np = test_u.nfree,length(get_free_dof_ids(test_p))
v0 = zeros(Nu+Np)
nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,v0,ode_cache_ok,v0)
bok = allocate_residual(nlop,v0)
Aok = allocate_jacobian(nlop,v0)

function gridap_quantities(x,k)
  xk = x[k]
  tk = times[k]
  ode_cache = nlop.ode_cache
  ode_op = nlop.odeop
  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,ode_op,tk)

  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  fill!(bok,z)
  residual!(bok,ode_op,tk,(xk,v0),ode_cache)
  jacobians!(Aok,ode_op,tk,(xk,v0),(1.0,1/(dt*θ)),ode_cache)
  return bok,Aok
end

# tests
x = copy(xn) #.* 0.
op = get_ptoperator(fesolver,_feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
_lhs = collect_lhs_contributions!(lhs_cache,info,op,_rblhs,rbspace)
_rhs = collect_rhs_contributions!(rhs_cache,info,op,_rbrhs,rbspace)

A,b = [],[]
for k = eachindex(times)
  bk,Ak = gridap_quantities(x,k)
  push!(b,copy(bk))
  push!(A,copy(Ak))
end

println("Norm res: $(map(norm,b))")
println("Norm jac: $(map(norm,A))")

LHS11 = NnzMatrix(NnzVector.(map(x->x[1:Nu,1:Nu],A))...)
LHS11_rb = space_time_projection(LHS11,rbspace[1],rbspace[1])
R1 = NnzMatrix(map(x->x[1:Nu],b)...)
RHS1_rb = space_time_projection(R1,rbspace[1])

norm(LHS11_rb - _lhs[1][1:nu,1:nu]) / norm(LHS11_rb)
norm(RHS1_rb - _rhs[1][1:nu]) / norm(RHS1_rb)
################################ CHECK NL MDEIM ################################
