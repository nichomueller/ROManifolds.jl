# THIS WORKS WELL!!
fun(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ)
_feop = PTFEOperator(fun,jac,jac_t,pspace,trial,test)
sols_mdeim,params_mdeim = sols[1:nsnaps_mdeim],params[1:nsnaps_mdeim]
op = get_ptoperator(fesolver,_feop,sols_mdeim,params_mdeim)
b = allocate_residual(op,sols_mdeim)
residual!(b,op,sols_mdeim)
nzm = NnzMatrix(b;nparams=nsnaps_mdeim)
basis_space,basis_time = compress(nzm;ϵ=rbinfo.ϵ)

# OK
sols_online,params_online = sols[end],params[end:end]
op_online = get_ptoperator(fesolver,_feop,sols_online,params_online)
b_online = allocate_residual(op_online,sols_online)
residual!(b_online,op_online,sols_online)
nzm_online = NnzMatrix(b_online;nparams=1)
err = nzm_online - basis_space*basis_space'*nzm_online

Φ = rbspace.basis_space
α = rand(size(Φ,2))
β = PTArray([Φ*α])
b_wrong = allocate_residual(op_online,sols_online)
residual!(b_wrong,op_online,sols_online)
nzm_wrong = NnzMatrix(b_wrong;nparams=1)
err = nzm_wrong - basis_space*basis_space'*nzm_wrong

# TRY WITH NAVIER STOKES
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

c(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
dc(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ) + ∫ₚ(v⊙(∇(u)'⋅du),dΩ)

res_lin(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
jac_lin(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)

T = Float
reffe_u = ReferenceFE(lagrangian,VectorValue{2,T},order)
reffe_p = ReferenceFE(lagrangian,T,order-1)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
trial_u = PTTrialFESpace(test_u,[g0,g])
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = PTMultiFieldFESpace([test_u,test_p])
trial = PTMultiFieldFESpace([trial_u,trial_p])
feop = PTFEOperator(res_lin,jac_lin,jac_t,(c,dc),pspace,trial,test)
t0,tf,dt,θ = 0.,0.05,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))

nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
fesolver = PThetaMethod(nls,xh0μ,θ,dt,t0,tf)

ϵ = [1e-4,1e-4]
load_solutions = true
save_solutions = true
load_structures = true
save_structures = true
norm_style = :l2
compute_supremizers = true
nsnaps_state = 50
nsnaps_mdeim = 30
nsnaps_test = 10
st_mdeim = false
rbinfo = BlockRBInfo(test_path;ϵ,norm_style,compute_supremizers,nsnaps_state,
  nsnaps_mdeim,nsnaps_test,st_mdeim)

sols,params = load(rbinfo,(BlockSnapshots{Vector{T}},Table))
rbspace = load(rbinfo,BlockRBSpace{T})
rbrhs,rblhs = load(rbinfo,(NTuple{2,BlockRBVecAlgebraicContribution{T}},
  NTuple{3,Vector{BlockRBMatAlgebraicContribution{T}}}),Ω)

# ARTIFICIAL MDEIM
function temp_rb_solver(rbinfo,feop,fesolver,rbspace,rbres,rbjacs,snaps,params)
  println("Solving nonlinear RB problems with Newton iterations")
  nsnaps_test = rbinfo.nsnaps_test
  snaps_train,params_train = snaps[1:nsnaps_test],params[1:nsnaps_test]
  snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
  x = nearest_neighbor(snaps_train,params_train,params_test)
  op = get_ptoperator(fesolver,feop,snaps_test,params_test)
  op_lin = linear_operator(op)
  op_nlin = nonlinear_operator(op)
  op_aux = auxiliary_operator(op)
  xrb = space_time_projection(x,rbspace)
  dxrb = similar(xrb)
  cache,(rhs,lhs) = allocate_cache(op,rbspace)
  newt_cache = nothing
  uh0_test = fesolver.uh0(params_test)
  conv0 = ones(nsnaps_test)
  rbrhs_lin,rbrhs_nlin = rbres
  rblhs_lin,rblhs_nlin,rblhs_aux = rbjacs
  xvec = []
  stats = @timed begin
    rhs_lin,(lhs_lin,lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op_lin,rbrhs_lin,rblhs_lin,rbspace)
    for iter in 1:fesolver.nls.max_nliters
      x = recenter(x,uh0_test;θ=fesolver.θ)
      push!(xvec,x)
      op_nlin = update_ptoperator(op_nlin,x)
      op_aux = update_ptoperator(op_aux,x)
      rhs_nlin,(lhs_nlin,) = collect_rhs_lhs_contributions!(cache,rbinfo,op_nlin,rbrhs_nlin,rblhs_nlin,rbspace)
      lhs_aux, = collect_lhs_contributions!(cache[2],rbinfo,op_aux,rblhs_aux,rbspace)
      @. lhs = lhs_lin+lhs_t+lhs_nlin
      @. rhs = rhs_lin+rhs_nlin+(lhs_lin+lhs_t+lhs_aux)*xrb
      newt_cache = rb_solve!(dxrb,fesolver.nls.ls,rhs,lhs,newt_cache)
      xrb += dxrb
      x = recast(xrb,rbspace)
      isconv,conv = Algebra._check_convergence(fesolver.nls,dxrb,conv0)
      println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
      if all(isconv); break; end
      if iter == fesolver.nls.max_nliters
        @unreachable
      end
    end
  end
  xvec
end

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
cc(μ,t,u,v) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
_feop = PTFEOperator(cc,jac_lin,jac_t,pspace,trial_u,test_u)
sols_mdeim,params_mdeim = sols[1:nsnaps_mdeim],params[1:nsnaps_mdeim]
# op = get_ptoperator(fesolver,_feop,sols_mdeim,params_mdeim)
op = get_ptoperator(fesolver,feop,rbspace,params_mdeim)
b = allocate_residual(op,sols_mdeim)
residual!(b,op,sols_mdeim)
nzm = NnzMatrix(b;nparams=nsnaps_mdeim)
basis_space,basis_time = compress(nzm)

xvec = temp_rb_solver(rbinfo,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)

params_test = params[end-nsnaps_test+1:end]
x = xvec[1]
op = get_ptoperator(fesolver,_feop,x,params_test)
b = allocate_residual(op,x)
residual!(b,op,x)
nzm = NnzMatrix(b;nparams=1)
err = nzm - basis_space*basis_space'*nzm

norm(err) / norm(nzm)

function new_nl_rb_solver(rbinfo,feop,fesolver,rbspace,rbres,rbjacs,snaps,params)
  println("Solving nonlinear RB problems with Newton iterations")
  nsnaps_test = rbinfo.nsnaps_test
  snaps_train,params_train = snaps[1:nsnaps_test],params[1:nsnaps_test]
  snaps_test,params_test = snaps[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
  x = nearest_neighbor(snaps_train,params_train,params_test)
  op = get_ptoperator(fesolver,feop,snaps_test,params_test)
  op_lin = linear_operator(op)
  op_nlin = nonlinear_operator(op)
  xrb = space_time_projection(x,rbspace)
  dxrb = similar(xrb)
  cache,(rhs,lhs) = allocate_cache(op,rbspace)
  newt_cache = nothing
  uh0_test = fesolver.uh0(params_test)
  conv0 = ones(nsnaps_test)
  rbrhs_lin,rbrhs_nlin = rbres
  rblhs_lin,rblhs_nlin = rbjacs
  stats = @timed begin
    rhs_lin,(lhs_lin,lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op_lin,rbrhs_lin,rblhs_lin,rbspace)
    for iter in 1:fesolver.nls.max_nliters
      x = recenter(x,uh0_test;θ=fesolver.θ)
      op_nlin = update_ptoperator(op_nlin,x)
      rhs_nlin,(lhs_nlin,) = collect_rhs_lhs_contributions!(cache,rbinfo,op_nlin,rbrhs_nlin,rblhs_nlin,rbspace)
      @. lhs = lhs_lin+lhs_t+lhs_nlin
      @. rhs = rhs_lin+rhs_nlin+(lhs_lin+lhs_t)*xrb
      newt_cache = rb_solve!(dxrb,fesolver.nls.ls,rhs,lhs,newt_cache)
      xrb += dxrb
      x = recast(xrb,rbspace)
      isconv,conv = Algebra._check_convergence(1e-4,dxrb,conv0)
      println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
      if all(isconv); break; end
      if iter == fesolver.nls.max_nliters
        @unreachable
      end
    end
  end
  post_process(rbinfo,feop,fesolver,snaps_test,params_test,x,stats)
end

Base.:(∘)(::Function,::Tuple{Vararg{Union{Nothing,CellField}}}) = nothing

jac_nl(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)
res_nl(μ,t,(u,p),(v,q)) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
# jac_nl(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ) + ∫ₚ(v⊙(∇(u)'⋅du),dΩ)
# res_nl(μ,t,(u,p),(v,q)) =  ∫ₚ(v⊙(∇(du)'⋅u),dΩ)
feop_nl = PTFEOperator(res_nl,jac_nl,jac_t,pspace,trial,test)
op_nlin = get_ptoperator(fesolver,feop_nl,rbspace,params_mdeim)
rhs_nlin = collect_compress_rhs(rbinfo,op_nlin,rbspace)
# lhs_nlin = collect_compress_lhs(rbinfo,op_nlin,rbspace)

rbrhs_lin,_ = rbrhs
rblhs_lin,lhs_nlin,_ = rblhs

new_rbrhs = rbrhs_lin,rhs_nlin
new_rblhs = rblhs_lin,lhs_nlin

new_nl_rb_solver(rbinfo,feop,fesolver,rbspace,new_rbrhs,new_rblhs,sols,params)

function get_rec_snaps(s::Snapshots{Vector{T}},rb::RBSpace{T},n::Int=1) where T
  rmat = map(1:n) do count
    mati = stack(s[count].array)
    project_recast(mati,rb)
  end
  array = Vector{T}[]
  for i = 1:n
    for j = 1:length(rmat[1])
      push!(array,rmat[i][j])
    end
  end
  PTArray(array)
end

function get_rec_snaps(s::BlockSnapshots,rb::BlockRBSpace,args...)
  map((si,bi) -> get_rec_snaps(si,bi,args...),s.blocks,rb.blocks)
end

snaps_mdeim_rec = get_rec_snaps(sols,rbspace,nsnaps_mdeim)
vsnaps_mdeim_rec = vcat(snaps_mdeim_rec...)
op_nlin = get_ptoperator(fesolver,feop_nl,vsnaps_mdeim_rec,params_mdeim)
rhs_nlin = collect_compress_rhs(rbinfo,op_nlin,rbspace)
lhs_nlin = collect_compress_lhs(rbinfo,op_nlin,rbspace)
rbrhs_lin,_ = rbrhs
rblhs_lin,_,_ = rblhs
new_rbrhs = rbrhs_lin,rhs_nlin
new_rblhs = rblhs_lin,lhs_nlin
new_nl_rb_solver(rbinfo,feop,fesolver,rbspace,new_rbrhs,new_rblhs,sols,params)
get_ptoperator(fesolver,feop,sols,params)

mati = stack(sols[1][1].array)
test_reduced_basis(mati,rbspace[1])
