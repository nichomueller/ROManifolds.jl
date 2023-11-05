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
  μ::Table)

  nsnaps_mdeim = info.nsnaps_mdeim
  θ = fesolver.θ

  _μ = μ[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,_μ)
  rhs = collect_compress_rhs(info,op,rbspace)
  lhs = collect_compress_lhs(info,op,rbspace;θ)
  show(rhs),show(lhs)

  rhs,lhs
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
rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,params)

snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
times = get_times(fesolver)
xn,μn = PTArray(snaps_test[1:length(times)]),params_test[1]
op = get_ptoperator(fesolver,feop,xn,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,xn)

nu = get_rb_ndofs(rbspace[1])
np = get_rb_ndofs(rbspace[2])

# THIS WORKS!!!!
################################## MDEIM #######################################
M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)
LHS11_2 = NnzMatrix([NnzVector(M/(θ*dt)) for _ = times]...)
LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
x = copy(xn) .* 0.
xrb = space_time_projection(x,op,rbspace)
op = get_ptoperator(fesolver,feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
for iter in 1:20
  rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
  dxrb = PTArray(sum(lhs)[1] \ (rhs[1] + vcat(LHS11_rb_2*xrb[1][1:nu],zeros(np))))
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
################################## MDEIM #######################################

################################ COMPARISON ####################################
g0_ok(x,t) = g0(x,μn,t)
g0_ok(t) = x->g0_ok(x,t)
g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ + ∫(v⊙(∇(u)'⋅du))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
Jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
Res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,u,v)
trial_u_ok = TransientTrialFESpace(test_u,[g0_ok,g_ok])
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(Res_ok,Jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)

times = get_times(fesolver)
kt = 1
t = times[kt]
v0 = zero(xn[1])
x = kt > 1 ? xn[kt-1] : get_free_dof_values(xh0μ(μn))
Nu,Np = test_u.nfree,length(get_free_dof_ids(test_p))
Nt = get_time_ndofs(fesolver)
θdt = θ*dt
vθ = zeros(Nu+Np)
ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,vθ,ode_cache_ok,vθ)
bok = allocate_residual(nlop0,vθ)
Aok = allocate_jacobian(nlop0,vθ)

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)

function return_quantities(ode_op,ode_cache,x::PTArray,kt)
  xk = x[kt]
  t = times[kt]
  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,ode_op,t)

  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  fill!(bok,z)
  residual!(bok,ode_op,t,(xk,v0),ode_cache)
  jacobians!(Aok,ode_op,t,(xk,v0),(1.0,1/(dt*θ)),ode_cache)
  return bok,Aok
end

x = copy(xn) .* 0.
nu = get_rb_ndofs(rbspace[1])

for iter in 1:fesolver.nls.max_nliters
  xrb = space_time_projection(map(x->x[1:Nu],x),rbspace[1]),space_time_projection(map(x->x[1+Nu:end],x),rbspace[2])
  A,b = [],[]
  for kt = eachindex(times)
    bk,Ak = return_quantities(ode_op_ok,ode_cache_ok,x,kt)
    push!(b,copy(bk))
    push!(A,copy(Ak))
  end

  LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/θdt),A)...)
  LHS11_2 = NnzMatrix([NnzVector(M/θdt) for _ = times]...)
  LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),A)...)
  LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),A)...)

  LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
  LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
  LHS11_rb = LHS11_rb_1 + LHS11_rb_2
  LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
  LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

  np = get_rb_ndofs(rbspace[2])
  LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

  R1 = NnzMatrix(map(x->x[1:Nu],b)...)
  R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
  RHS1_rb = space_time_projection(R1,rbspace[1])
  RHS2_rb = space_time_projection(R2,rbspace[2])
  _RHS_rb = vcat(RHS1_rb,RHS2_rb)
  RHS_rb = vcat(LHS11_rb_2*vcat(xrb...)[1][1:nu],zeros(np)) + _RHS_rb

  op = update_ptoperator(op,x)
  rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
  lhs,lhs_t = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)

  println("Norm jac: $(norm(LHS_rb - (lhs+lhs_t)[1]))")
  println("Norm res: $(norm(RHS_rb - (rhs+lhs_t*vcat(xrb...))[1]))")

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

############################# LIKE OLD CODE ####################################
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

Base.getindex(::Nothing,::GenericMeasure) = nothing

# linear part (Stokes)
res_lin(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
jac_lin(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
jac_t_lin(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
feop_lin = AffinePTFEOperator(res_lin,jac_lin,jac_t_lin,pspace,trial,test)
rbrhs_lin,rblhs_lin = collect_compress_rhs_lhs(info,feop_lin,fesolver,rbspace,params)

# nonlinear part
_c(u,du,v) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
dir(μ,t) = zero(trial_u(μ,t))
res_nlin(μ,t,(u,p),(v,q)) = _c(u,dir(μ,t),v)
jac_nlin(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v)
jac_t_nlin(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
feop_nlin = PTFEOperator(res_nlin,jac_nlin,jac_t_nlin,pspace,trial,test)
rbrhs_nlin,rblhs_nlin = collect_compress_rhs_lhs(info,feop_nlin,fesolver,rbspace,params)

# aux part
res_aux(μ,t,(u,p),(v,q)) = nothing
jac_aux(μ,t,(u,p),(du,dp),(v,q)) = _c(u,du,v)
jac_t_aux(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
feop_aux = PTFEOperator(res_aux,jac_aux,jac_t_aux,pspace,trial,test)
rblhs_aux = collect_compress_lhs(info,feop_aux,fesolver,rbspace,params)

x = copy(xn) .* 0.
op_lin = get_ptoperator(fesolver,feop_lin,x,Table([μn]))
op_nlin = get_ptoperator(fesolver,feop_nlin,x,Table([μn]))
op_aux = get_ptoperator(fesolver,feop_aux,x,Table([μn]))
xrb = space_time_projection(x,op_lin,rbspace)
rhs_cache,lhs_cache = allocate_cache(op_lin,x)
# THIS WORKS!!!!!!
for iter in 1:20
  rhs_lin = collect_rhs_contributions!(rhs_cache,info,op_lin,rbrhs_lin,rbspace)
  lhs_lin,lhs_t = collect_lhs_contributions!(lhs_cache,info,op_lin,rblhs_lin,rbspace)
  rhs_nlin = collect_rhs_contributions!(rhs_cache,info,op_nlin,rbrhs_nlin,rbspace)
  lhs_nlin, = collect_lhs_contributions!(lhs_cache,info,op_nlin,rblhs_nlin,rbspace)
  lhs_aux, = collect_lhs_contributions!(lhs_cache,info,op_aux,rblhs_aux,rbspace)
  lhs = lhs_lin+lhs_t+lhs_nlin
  rhs = rhs_lin+rhs_nlin+(lhs_lin+lhs_t+lhs_aux)*xrb
  dxrb = PTArray(lhs[1] \ rhs[1])
  xrb -= dxrb
  x = recast(xrb,rbspace)
  op_lin = update_ptoperator(op_lin,x)
  op_nlin = update_ptoperator(op_nlin,x)
  op_aux = update_ptoperator(op_aux,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))

################################################################################
# linear part (Stokes)
res_lin(μ,t,(u,p),(v,q)) = nothing
jac_lin(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
jac_t_lin(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
feop_lin = AffinePTFEOperator(res_lin,jac_lin,jac_t_lin,pspace,trial,test)
rblhs_lin = collect_compress_lhs(info,feop_lin,fesolver,rbspace,params)

# nonlinear part
_c(u,du,v) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
dir(μ,t) = zero(trial_u(μ,t))
res_nlin(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ) +_c(u,dir(μ,t),v)
jac_nlin(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v)
jac_t_nlin(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
feop_nlin = PTFEOperator(res_nlin,jac_nlin,jac_t_nlin,pspace,trial,test)
rbrhs_nlin,rblhs_nlin = collect_compress_rhs_lhs(info,feop_nlin,fesolver,rbspace,params)

# aux part
res_aux(μ,t,(u,p),(v,q)) = nothing
jac_aux(μ,t,(u,p),(du,dp),(v,q)) = _c(u,du,v)
jac_t_aux(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
feop_aux = PTFEOperator(res_aux,jac_aux,jac_t_aux,pspace,trial,test)
rblhs_aux = collect_compress_lhs(info,feop_aux,fesolver,rbspace,params)

x = copy(xn) .* 0.
op_lin = get_ptoperator(fesolver,feop_lin,x,Table([μn]))
op_nlin = get_ptoperator(fesolver,feop_nlin,x,Table([μn]))
op_aux = get_ptoperator(fesolver,feop_aux,x,Table([μn]))
xrb = space_time_projection(x,op_lin,rbspace)
rhs_cache,lhs_cache = allocate_cache(op_lin,x)

for iter in 1:20
  lhs_lin,lhs_t = collect_lhs_contributions!(lhs_cache,info,op_lin,rblhs_lin,rbspace)
  rhs_nlin = collect_rhs_contributions!(rhs_cache,info,op_nlin,rbrhs_nlin,rbspace)
  lhs_nlin, = collect_lhs_contributions!(lhs_cache,info,op_nlin,rblhs_nlin,rbspace)
  lhs_aux, = collect_lhs_contributions!(lhs_cache,info,op_aux,rblhs_aux,rbspace)
  lhs = lhs_lin+lhs_t+lhs_nlin
  rhs = rhs_nlin+(lhs_t+lhs_aux)*xrb
  dxrb = PTArray(lhs[1] \ rhs[1])
  xrb -= dxrb
  x = recast(xrb,rbspace)
  op_lin = update_ptoperator(op_lin,x)
  op_nlin = update_ptoperator(op_nlin,x)
  op_aux = update_ptoperator(op_aux,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))

############################## SHORTER STRATEGY ################################
# PART1
ddir(μ,t) = zero(∂ₚt(trial_u)(μ,t))
res1(μ,t,(u,p),(v,q)) = (∫ₚ(v⋅ddir(μ,t),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(dir(μ,t)),dΩ)
  + _c(u,dir(μ,t),v) - ∫ₚ(zero(trial_p)*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(dir(μ,t))),dΩ))
jac1(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) + dc(u,du,v) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
jac_t1(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
feop1 = PTFEOperator(res1,jac1,jac_t1,pspace,trial,test)
rbrhs1,rblhs1 = collect_compress_rhs_lhs(info,feop1,fesolver,rbspace,params)

# PART2
res2(μ,t,(u,p),(v,q)) = nothing
jac2(μ,t,(u,p),(du,dp),(v,q)) = _c(u,du,v)
jac_t2(μ,t,(u,p),(dut,dpt),(v,q)) = nothing
feop2 = PTFEOperator(res2,jac2,jac_t2,pspace,trial,test)
rblhs2 = collect_compress_lhs(info,feop2,fesolver,rbspace,params)

x = copy(xn) .* 0.
op1 = get_ptoperator(fesolver,feop1,x,Table([μn]))
op2 = get_ptoperator(fesolver,feop2,x,Table([μn]))
xrb = space_time_projection(x,op1,rbspace)
rhs_cache,lhs_cache = allocate_cache(op1,x)

for iter in 1:20
  rhs1 = collect_rhs_contributions!(rhs_cache,info,op1,rbrhs1,rbspace)
  lhs1,lhs_t = collect_lhs_contributions!(lhs_cache,info,op1,rblhs1,rbspace)
  lhs2, = collect_lhs_contributions!(lhs_cache,info,op2,rblhs2,rbspace)
  lhs = lhs1+lhs_t
  rhs = rhs1+(lhs2+lhs_t)*xrb
  dxrb = PTArray(lhs[1] \ rhs[1])
  xrb -= dxrb
  x = recast(xrb,rbspace)
  op1 = update_ptoperator(op1,x)
  op2 = update_ptoperator(op2,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))
