# using Gridap
# using GridapDistributed
# using PartitionedArrays

# function main(ranks)
#   domain = (0,1,0,1)
#   mesh_partition = (2,2)
#   mesh_cells = (4,4)
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 2
#   u((x,y)) = (x+y)^order
#   f(x) = -Δ(u,x)
#   reffe = ReferenceFE(lagrangian,Float64,order)
#   V = TestFESpace(model,reffe,dirichlet_tags="boundary")
#   U = TrialFESpace(u,V)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,2*order)
#   a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
#   l(v) = ∫( v*f )dΩ
#   op = AffineFEOperator(a,l,U,V)
#   uh = solve(op)
#   writevtk(Ω,"results",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
# end

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   main(ranks)
# end

using LinearAlgebra
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using GridapDistributed
using PartitionedArrays
using Mabla.FEM
using Mabla.RB
using Mabla.Distributed

root = pwd()
test_path = "$root/results/HeatEquation/cube_2x2.json"
ϵ = 1e-4
load_solutions = true
save_solutions = true
load_structures = false
save_structures = true
postprocess = true
norm_style = :H1
nsnaps_state = 50
nsnaps_mdeim = 20
nsnaps_test = 10
st_mdeim = true
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

#   order = 1
#   degree = 2*order
#   Ω = Triangulation(model)
#   Γn = BoundaryTriangulation(model,tags=[7,8])
#   dΩ = Measure(Ω,degree)
#   dΓn = Measure(Γn,degree)

#   ranges = fill([1.,10.],3)
#   pspace = PSpace(ranges)

#   a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
#   a(μ,t) = x->a(x,μ,t)
#   aμt(μ,t) = PTFunction(a,μ,t)

#   f(x,μ,t) = 1.
#   f(μ,t) = x->f(x,μ,t)
#   fμt(μ,t) = PTFunction(f,μ,t)

#   h(x,μ,t) = abs(cos(t/μ[3]))
#   h(μ,t) = x->h(x,μ,t)
#   hμt(μ,t) = PTFunction(h,μ,t)

#   g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
#   g(μ,t) = x->g(x,μ,t)

#   u0(x,μ) = 0
#   u0(μ) = x->u0(x,μ)
#   u0μ(μ) = PFunction(u0,μ)

#   res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
#   jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
#   jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialPFESpace(test,g)
#   feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
#   t0,tf,dt,θ = 0.,0.3,0.005,0.5
#   uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
#   fe = trial(Table([rand(3) for _ = 1:3]),[t0,dt])
#   println(typeof(fe))
#   fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
#   uh0μ(Table([rand(3) for _ = 1:3]))
#   error("stop")

#   sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
#   # rbspace = reduced_basis(rbinfo,feop,sols)
#   # rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
# end

ranks = LinearIndices((4,))
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
order = 1
degree = 2*order
Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

ranges = fill([1.,10.],3)
pspace = PSpace(ranges)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = PTFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = PTFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)

# res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
# jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
# jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)
res(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

nparams = 2
params = realization(feop,nparams)
w0 = get_free_dof_values(uh0μ(params))
time_ndofs = num_time_dofs(fesolver)
T = get_vector_type(feop.test)
sol = PODESolution(fesolver,feop,params,w0,t0,tf)
Base.iterate(sol)

function Arrays.return_value(f::Broadcasting,x...)
  println(typeof(x[2]))
  broadcast( (y...) -> f.f(testargs(f.f,y...)...), x... )
end

μ = params
ode_cache = allocate_cache(feop,μ,t0)
vθ = similar(w0)
vθ .= 0.0
ode_cache = update_cache!(ode_cache,feop,μ,t0)
lop = PTAffineThetaMethodOperator(feop,μ,t0,dt*θ,w0,ode_cache,vθ)
Xh, = ode_cache
uh = EvaluationFunction(Xh[1],vθ)
dxh = ()
for _ in 1:get_order(lop.feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
∫(dv*∂ₚt(xh))dΩ
∫(aμt(μ,dt)*∇(dv)⋅∇(xh))dΩ
∫(fμt(μ,dt)*dv)dΩ
∫(hμt(μ,dt)*dv)dΓn
# evaluate!(nothing,Operation(*),dv.fields[1],xh.fields[1])
# fields = map(aa.fields,b.fields) do f,g
#   evaluate!(nothing,k,f,g)
# end
cf = (aμt(μ,dt)*∇(dv)⋅∇(xh)).fields[1]
quad = dΩ.measures[1].quad
# b = change_domain(cf,quad.trian,quad.data_domain_style)
# change_domain(cf.args[2],quad.trian,quad.data_domain_style)
strian = get_triangulation(cf.args[2])
ttrian = quad.trian
# quad.trian === get_triangulation(cf.args[2])
D = num_cell_dims(strian)
sglue = get_glue(strian,Val(D))
tglue = get_glue(ttrian,Val(D))
# CellData.change_domain_ref_ref(cf.args[2],ttrian,sglue,tglue)
sface_to_field = get_data(cf.args[2])
mface_to_sface = sglue.mface_to_tface
tface_to_mface = tglue.tface_to_mface
tface_to_mface_map = tglue.tface_to_mface_map
mface_to_field = extend(sface_to_field,mface_to_sface)
tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
# tface_to_field_t = lazy_map(Broadcasting(∘),tface_to_field_s,tface_to_mface_map)
fi = map(testitem,(tface_to_field_s,tface_to_mface_map))
return_value(Broadcasting(∘),fi...)
broadcast((y...) ->(∘)(testargs(∘,y...)...),fi...)

# times = get_times(fesolver)
# dv = get_fe_basis(test)
# x = sol.u0
# Xh, = Gridap.ODEs.TransientFETools.allocate_cache(feop,params,times)
# xh = TransientCellField(EvaluationFunction(Xh[1],x),(EvaluationFunction(Xh[1],x),))
# dc = integrate(op.res(μ,t,xh,dv))

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 1

#   g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
#   g(μ,t) = x->g(x,μ,t)
#   u0(x,μ) = 0
#   u0(μ) = x->u0(x,μ)
#   u0μ(μ) = PFunction(u0,μ)

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialPFESpace(test,g)
#   t0 = 0.
#   uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
#   uh0μ(Table([rand(3) for _ = 1:3]))
# end

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 1

#   g(x,t) = exp(-x[1])*abs(sin(t))
#   g(t) = x->g(x,t)
#   u0(x) = 0

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialFESpace(test,g)
#   t0 = 0.
#   uh0μ = interpolate_everywhere(u0,trial(t0))
# end

μ = params
test_g(x,t) = exp(-x[1])*abs(sin(t))
test_g(t) = x->test_g(x,t)
test_res(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μ[1],t)*∇(v)⋅∇(u))dΩ - ∫(f(μ[1],t)*v)dΩ - ∫(h(μ[1],t)*v)dΓn
test_jac(t,u,du,v) = ∫(a(μ[1],t)*∇(v)⋅∇(du))dΩ
test_jac_t(t,u,dut,v) = ∫(v*dut)dΩ

test_trial = TransientTrialFESpace(test,test_g)
test_feop = TransientFEOperator(test_res,test_jac,test_jac_t,test_trial,test)
test_xh0 = interpolate_everywhere(u0(μ[1]),test_trial(t0))
ode_solver = ThetaMethod(LUSolver(),dt,θ)
sol_t = solve(ode_solver,test_feop,test_xh0,t0,tf)
Base.iterate(sol_t)

test_w0 = get_free_dof_values(test_xh0)
test_vθ = similar(test_w0)
test_vθ .= 0.0
test_op = get_algebraic_operator(test_feop)
test_ode_cache = Gridap.ODEs.TransientFETools.allocate_cache(test_op)
test_ode_cache = update_cache!(test_ode_cache,test_op,t0)
test_Xh, = test_ode_cache
test_xh = TransientCellField(EvaluationFunction(test_Xh[1],test_vθ),(EvaluationFunction(test_Xh[1],test_vθ),))
test_du = get_trial_fe_basis(test_trial(nothing))
∫(∇(dv)⋅∇(test_xh))dΩ

test_cf = dv*test_xh
test_sface_to_field = get_data(test_cf.args[2])
mface_to_sface = sglue.mface_to_tface
tface_to_mface = tglue.tface_to_mface
tface_to_mface_map = tglue.tface_to_mface_map
test_mface_to_field = extend(test_sface_to_field,mface_to_sface)
test_tface_to_field_s = lazy_map(Reindex(test_mface_to_field),tface_to_mface)
test_tface_to_field_t = lazy_map(Broadcasting(∘),test_tface_to_field_s,tface_to_mface_map)
test_fi = map(testitem,(test_tface_to_field_s,tface_to_mface_map))
return_value(Broadcasting(∘),test_fi...)
broadcast((y...) ->(∘)(testargs(∘,y...)...),test_fi...)

function FESpaces.change_domain(a::CellField,strian::Triangulation,::ReferenceDomain,ttrian::Triangulation,::ReferenceDomain)
  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """
  if strian === ttrian
    return a
  end

  @assert is_change_possible(strian,ttrian) msg
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  CellData.change_domain_ref_ref(a,ttrian,sglue,tglue)
end

f1 = xh.cellfield.fields[1]
test_f1 = test_xh.cellfield.fields[1]

f1.cell_dof_values
test_f1.cell_dof_values

for k in eachindex(f1.cell_dof_values)
  @assert test_f1.cell_dof_values[k] == f1.cell_dof_values[k][1]
  @assert test_f1.dirichlet_values[k] == f1.dirichlet_values[k][1]
  @assert test_f1.free_values[k] == f1.free_values[k][1]
end

##########
model = CartesianDiscreteModel(domain,mesh_cells)
order = 1
degree = 2*order
Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)
T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

nparams = 2
params = realization(feop,nparams)
w0 = get_free_dof_values(uh0μ(params))

ode_cache = allocate_cache(feop,μ,t0)
vθ = similar(w0)
vθ .= 0.0
ode_cache = update_cache!(ode_cache,feop,μ,t0)
lop = PTAffineThetaMethodOperator(feop,μ,t0,dt*θ,w0,ode_cache,vθ)
Xh, = ode_cache
uh = EvaluationFunction(Xh[1],vθ)
dxh = ()
for _ in 1:get_order(lop.feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)
cf = dv*xh
∫(dv*xh)dΩ
quad = dΩ.quad
strian = get_triangulation(cf.args[2])
ttrian = quad.trian
D = num_cell_dims(strian)
sglue = get_glue(strian,Val(D))
tglue = get_glue(ttrian,Val(D))
sface_to_field = get_data(cf.args[2])
mface_to_sface = sglue.mface_to_tface
tface_to_mface = tglue.tface_to_mface
tface_to_mface_map = tglue.tface_to_mface_map
mface_to_field = extend(sface_to_field,mface_to_sface)
tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
fi = map(testitem,(tface_to_field_s,tface_to_mface_map))
return_value(Broadcasting(∘),fi...)
broadcast((y...) ->(∘)(testargs(∘,y...)...),fi...)

(∘)(fi...)

# Base.:∘(f::PTArray{<:Field},g::Field) = map(f->Operation(f)(g),f)
# Base.:∘(f::Field,g::PTArray{<:Field}) = map(g->Operation(f)(g),g)

# function Arrays.evaluate!(
#   cache,
#   ::Broadcasting{typeof(∘)},
#   f::PTArray{<:Field},
#   g::Field)

#   map(f) do f
#     f∘g
#   end
# end

# function Arrays.evaluate!(
#   cache,
#   ::Broadcasting{typeof(∘)},
#   f::Field,
#   g::PTArray{<:Field})

#   map(g) do g
#     f∘g
#   end
# end
