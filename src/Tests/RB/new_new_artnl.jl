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

get_residual(op::TempPTFEOperator) = op.res
get_jacobian(op::TempPTFEOperator) = op.jacs

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

sols,params = load(info,(BlockSnapshots,Table))
rbspace = load(info,BlockRBSpace)
rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,params)

snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
times = get_times(fesolver)
xn,μn = PTArray(vcat(snaps_test...)[1:length(times)]),params_test[1]
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
for iter in 1:fesolver.nls.max_nliters
  rhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
  dxrb = PTArray(lhs[1] \ (rhs[1] + vcat(LHS11_rb_2*xrb[1][1:nu],zeros(np))))
  xrb -= dxrb
  x = vcat(recast(xrb,rbspace)...)
  op = update_ptoperator(op,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))
################################## MDEIM #######################################

############################### ACTUAL MDEIM ###################################
begin
  _c(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
  _dc(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ) + ∫ₚ(v⊙(∇(u)'⋅du),dΩ)

  _res(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  _jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
  _jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)

  _feop = NonlinearPTFEOperator(_res,_jac,_jac_t,(_c,_dc),pspace,trial,test)
end

sols,params = load(info,(BlockSnapshots,Table))
rbspace = load(info,BlockRBSpace)
_rbrhs,_rblhs,_nl_rbrhs,_nl_rblhs = collect_compress_rhs_lhs(info,_feop,fesolver,rbspace,params)
_x = copy(xn) .* 0.
_op = get_ptoperator(fesolver,_feop,_x,Table([μn]))
_xrb = space_time_projection(_x,_op,rbspace)
_rhs_cache,_lhs_cache = allocate_cache(_op,xn)
for iter in 1:fesolver.nls.max_nliters
  _lrhs = collect_rhs_contributions!(_rhs_cache,info,_op,_rbrhs,rbspace)
  _llhs = collect_lhs_contributions!(_lhs_cache,info,_op,_rblhs,rbspace)
  _nlrhs = collect_rhs_contributions!(_rhs_cache,info,_op,_nl_rbrhs,rbspace)
  _nllhs = collect_lhs_contributions!(_lhs_cache,info,_op,_nl_rblhs,rbspace)
  _lhs = _llhs + _nllhs
  _rhs = _llhs*_xrb + (_lrhs + _nlrhs) #
  _dxrb = NonaffinePTArray([_lhs[1] \ _rhs[1]])
  _xrb -= _dxrb
  _x = vcat(recast(_xrb,rbspace)...)
  _op = update_ptoperator(_op,_x)
  nerr = norm(_dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(_x.array...)) / norm(hcat(xn.array...))
############################### ACTUAL MDEIM ###################################

################################ COMPARISON ####################################
x = copy(xn) .* 0.
xrb = space_time_projection(x,op,rbspace)
op = get_ptoperator(fesolver,feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
_op = get_ptoperator(fesolver,_feop,x,Table([μn]))
_rhs_cache,_lhs_cache = allocate_cache(_op,xn)
for iter in 1:fesolver.nls.max_nliters
  rhs_temp = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
  rhs = rhs_temp + NonaffinePTArray([vcat(LHS11_rb_2*xrb[1][1:nu],zeros(np))])
  lhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
  _lrhs = collect_rhs_contributions!(_rhs_cache,info,_op,_rbrhs,rbspace)
  _llhs = collect_lhs_contributions!(_lhs_cache,info,_op,_rblhs,rbspace)
  _nlrhs = collect_rhs_contributions!(_rhs_cache,info,_op,_nl_rbrhs,rbspace)
  _nllhs = collect_lhs_contributions!(_lhs_cache,info,_op,_nl_rblhs,rbspace)
  _lhs = _llhs + _nllhs
  _rhs = _lrhs + _nlrhs + _llhs*xrb#- _llhs*xrb
  println("Err jac: $(norm(lhs[1] - _lhs[1])/norm(lhs[1]))")
  println("Err res: $(norm(rhs[1] - _rhs[1])/norm(rhs[1]))")
  dxrb = NonaffinePTArray([lhs[1] \ rhs[1]])
  xrb -= dxrb
  x = vcat(recast(xrb,rbspace)...)
  op = update_ptoperator(op,x)
  nerr = norm(dxrb[1])
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))

x = copy(xn) .* 0.
xrb = space_time_projection(x,op,rbspace)
op = get_ptoperator(fesolver,feop,x,Table([μn]))
rhs_cache,lhs_cache = allocate_cache(op,x)
rhs_temp = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)[1]
rhs = rhs_temp + vcat(LHS11_rb_2*xrb[1][1:nu],zeros(np))
lhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)[1]

_rbrhs,_rblhs,_nl_rbrhs,_nl_rblhs = collect_compress_rhs_lhs(info,_feop,fesolver,rbspace,params)
_xrb = space_time_projection(x,_op,rbspace)
_op = get_ptoperator(fesolver,_feop,x,Table([μn]))
_rhs_cache,_lhs_cache = allocate_cache(_op,xn)
_lrhs = collect_rhs_contributions!(_rhs_cache,info,_op,_rbrhs,rbspace)[1]
_llhs = collect_lhs_contributions!(_lhs_cache,info,_op,_rblhs,rbspace)[1]
_nlrhs = collect_rhs_contributions!(_rhs_cache,info,_op,_nl_rbrhs,rbspace)[1]
_nllhs = collect_lhs_contributions!(_lhs_cache,info,_op,_nl_rblhs,rbspace)[1]
_lhs = _llhs + _nllhs
_rhs = _llhs*_xrb[1] + (_lrhs + _nlrhs)

lhs - _lhs
rhs - _rhs
################################ COMPARISON ####################################
